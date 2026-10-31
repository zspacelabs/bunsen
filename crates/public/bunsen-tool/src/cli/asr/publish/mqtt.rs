//! The `mqtt` target: each record published to `{prefix}/{event}` on a broker.
use std::{
    sync::{
        Arc,
        atomic::{
            AtomicBool,
            AtomicUsize,
            Ordering,
        },
    },
    thread::JoinHandle,
    time::{
        Duration,
        Instant,
    },
};

use bunsen::errors::{
    BunsenError,
    BunsenResult,
};
use rumqttc::{
    Client,
    Connection,
    Event as MqttEvent,
    MqttOptions,
    Outgoing,
    Packet,
    QoS,
    RecvTimeoutError,
};

use super::{
    Cx,
    Event,
    Opts,
    Sink,
    config::MqttConfig,
    record::json_line,
};

/// Requests the client queues before a publish is dropped.
const QUEUE: usize = 1024;

/// How often the connection's thread looks up from the broker.
const POLL: Duration = Duration::from_millis(100);

/// How long [`finish`](Sink::finish) waits for the broker to take what is
/// queued.
const DRAIN: Duration = Duration::from_secs(5);

/// The default broker port.
const PORT: u16 = 1883;

/// `[mqtt://|tcp://]HOST[:PORT]`'s host and port.
fn parse_broker(broker: &str) -> BunsenResult<(String, u16)> {
    let rest = match broker.split_once("://") {
        None => broker,
        Some(("mqtt" | "tcp", rest)) => rest,
        Some((scheme, _)) => {
            return Err(BunsenError::policy(format!(
                "mqtt broker {broker}: `{scheme}://` is not supported; `mqtt://HOST[:PORT]`"
            )));
        }
    };
    let rest = rest.trim_end_matches('/');
    let (host, port) = match rest.rsplit_once(':') {
        Some((host, port)) => (
            host,
            port.parse().map_err(|_| {
                BunsenError::policy(format!("mqtt broker {broker}: bad port {port:?}"))
            })?,
        ),
        None => (rest, PORT),
    };
    if host.is_empty() {
        return Err(BunsenError::policy(format!(
            "mqtt broker {broker}: no host"
        )));
    }
    Ok((host.to_string(), port))
}

/// Publishes each record to `{prefix}/{event}` on a broker, a JSON payload,
/// at the configured `QoS`.
///
/// The connection runs on a thread of its own, reconnecting as `rumqttc`
/// does; a publish is queued without waiting, and dropped, logged, while
/// the queue is full. [`finish`](Sink::finish) waits up to 5 s for the
/// broker to take (`QoS` 0) or acknowledge (`QoS` 1 and 2) every queued
/// publish, then disconnects.
pub struct MqttSink {
    client: Client,
    prefix: String,
    qos: QoS,
    retain: bool,
    queued: Arc<AtomicUsize>,
    done: Arc<AtomicBool>,
    worker: JoinHandle<()>,
}

impl MqttSink {
    /// Connects to `config`'s broker.
    ///
    /// # Errors
    /// [`Policy`](bunsen::errors::BunsenErrorKind::Policy) for a broker that
    /// does not parse, or a `QoS` past 2.
    pub fn open(config: MqttConfig) -> BunsenResult<Self> {
        let (host, port) = parse_broker(&config.broker)?;
        let qos = match config.qos {
            0 => QoS::AtMostOnce,
            1 => QoS::AtLeastOnce,
            2 => QoS::ExactlyOnce,
            n => return Err(BunsenError::policy(format!("mqtt qos {n}: 0, 1 or 2"))),
        };
        let id = config
            .client_id
            .unwrap_or_else(|| format!("bunsen-tool-{}", std::process::id()));
        let mut options = MqttOptions::new(id, &host, port);
        options.set_keep_alive(Duration::from_secs(30));
        if let Some(username) = config.username {
            options.set_credentials(username, config.password.unwrap_or_default());
        }
        let (client, connection) = Client::new(options, QUEUE);
        let queued = Arc::new(AtomicUsize::new(0));
        let done = Arc::new(AtomicBool::new(false));
        let worker = {
            let (client, queued, done) = (client.clone(), queued.clone(), done.clone());
            let broker = format!("{host}:{port}");
            std::thread::spawn(move || drive(connection, client, qos, &broker, &queued, &done))
        };
        Ok(Self {
            client,
            prefix: config.prefix.trim_end_matches('/').to_string(),
            qos,
            retain: config.retain,
            queued,
            done,
            worker,
        })
    }
}

/// Runs `connection` until the sink is done and the broker has what was
/// queued (or [`DRAIN`] has passed), then disconnects.
fn drive(
    mut connection: Connection,
    client: Client,
    qos: QoS,
    broker: &str,
    queued: &AtomicUsize,
    done: &AtomicBool,
) {
    let mut delivered = 0usize;
    let mut failing = false;
    let mut deadline: Option<Instant> = None;
    let mut disconnecting = false;
    loop {
        match connection.recv_timeout(POLL) {
            Ok(Ok(event)) => {
                if failing {
                    log::warn!("mqtt {broker}: connected again");
                    failing = false;
                }
                match event {
                    MqttEvent::Incoming(Packet::ConnAck(_)) => {
                        log::info!("mqtt {broker}: connected");
                    }
                    MqttEvent::Outgoing(Outgoing::Publish(_)) if qos == QoS::AtMostOnce => {
                        delivered += 1;
                    }
                    MqttEvent::Incoming(Packet::PubAck(_) | Packet::PubComp(_)) => {
                        delivered += 1;
                    }
                    MqttEvent::Outgoing(Outgoing::Disconnect) => break,
                    _ => {}
                }
            }
            Ok(Err(e)) => {
                if !failing {
                    log::warn!("mqtt {broker}: {e}");
                    failing = true;
                }
                if deadline.is_some_and(|d| Instant::now() > d) {
                    break;
                }
                // `rumqttc` reconnects on the next poll; not at once.
                std::thread::sleep(Duration::from_secs(1));
            }
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => break,
        }
        if done.load(Ordering::SeqCst) && !disconnecting {
            let deadline = *deadline.get_or_insert_with(|| Instant::now() + DRAIN);
            let queued = queued.load(Ordering::SeqCst);
            let late = Instant::now() > deadline;
            if delivered >= queued || late {
                if delivered < queued {
                    log::warn!(
                        "mqtt {broker}: {} of {queued} records not delivered",
                        queued - delivered
                    );
                }
                disconnecting = true;
                if client.try_disconnect().is_err() || failing {
                    break;
                }
            }
        }
        if disconnecting && deadline.is_some_and(|d| Instant::now() > d + DRAIN) {
            break;
        }
    }
}

impl Sink for MqttSink {
    fn push(
        &mut self,
        event: &Event,
        cx: &Cx,
        opts: &Opts,
    ) -> BunsenResult<()> {
        let topic = format!("{}/{}", self.prefix, event.kind().name());
        match self
            .client
            .try_publish(topic, self.qos, self.retain, json_line(event, opts, cx))
        {
            Ok(()) => {
                self.queued.fetch_add(1, Ordering::SeqCst);
            }
            Err(e) => log::warn!("mqtt: a record dropped: {e}"),
        }
        Ok(())
    }

    fn finish(self: Box<Self>) -> BunsenResult<()> {
        self.done.store(true, Ordering::SeqCst);
        self.worker
            .join()
            .map_err(|_| BunsenError::sys("the mqtt worker panicked"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_broker() {
        assert_eq!(
            parse_broker("localhost").unwrap(),
            ("localhost".into(), 1883)
        );
        assert_eq!(
            parse_broker("mqtt://broker.lan:1884/").unwrap(),
            ("broker.lan".into(), 1884)
        );
        assert_eq!(
            parse_broker("tcp://10.0.0.2").unwrap(),
            ("10.0.0.2".into(), 1883)
        );
        assert!(parse_broker("mqtts://broker").is_err());
        assert!(parse_broker("broker:port").is_err());
        assert!(parse_broker("mqtt://").is_err());
    }
}
