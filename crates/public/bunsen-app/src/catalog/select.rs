//! `select`: its flags, as stages in command-line order, and its report.

use super::*;

/// The `select` flags: see [`CatalogAction::Select`].
#[derive(clap::Args, Clone, Debug)]
pub struct SelectArgs {
    /// The kind of entry to select.
    ///
    /// `build`, `template_class`, `namespace`, `resource`, ..
    #[arg(long, default_value = "build")]
    kind: String,
    /// Names, in order: one `from` stage; a named selection's name is a
    /// `use`.
    names: Vec<String>,
    /// The query's floor.
    #[arg(long, value_name = "F")]
    floor: Option<String>,
    #[command(flatten)]
    stages: SelectionStageArgs,
    /// The defaults, a list of stages: applied unless the only name is one
    /// build.
    ///
    /// A file, or YAML flow.
    #[arg(long, value_name = "FILE|[..]")]
    defaults: Option<String>,
    /// Also list every row a stage removed, with why.
    #[arg(long)]
    rank: bool,
}

/// The pipeline's stage and preset flags, for `#[command(flatten)]` into any
/// command that chooses builds: each flag is one stage, in command-line
/// order, and `--preset`, `--on` and `--off` go into the prefix.
///
/// The values are read back in order from the command's [`ArgMatches`] by
/// [`SelectionFlags::read`]; the fields only declare the flags.
#[derive(clap::Args, Clone, Debug, Default)]
pub struct SelectionStageArgs {
    /// `keep: P` (a name or a flow map); repeatable.
    #[arg(long, value_name = "P")]
    keep: Vec<String>,
    /// `drop: P`; repeatable.
    #[arg(long, value_name = "P")]
    drop: Vec<String>,
    /// `keep: {KEY: {lt: V}}`; repeatable.
    #[arg(long, num_args = 2, value_names = ["KEY", "V"])]
    lt: Vec<String>,
    /// `keep: {KEY: {le: V}}`; repeatable.
    #[arg(long, num_args = 2, value_names = ["KEY", "V"])]
    le: Vec<String>,
    /// `keep: {KEY: {gt: V}}`; repeatable.
    #[arg(long, num_args = 2, value_names = ["KEY", "V"])]
    gt: Vec<String>,
    /// `keep: {KEY: {ge: V}}`; repeatable.
    #[arg(long, num_args = 2, value_names = ["KEY", "V"])]
    ge: Vec<String>,
    /// `keep: {KEY: {eq: V}}`; repeatable.
    #[arg(long, num_args = 2, value_names = ["KEY", "V"])]
    eq: Vec<String>,
    /// `keep: {KEY: {ne: V}}`; repeatable.
    #[arg(long, num_args = 2, value_names = ["KEY", "V"])]
    ne: Vec<String>,
    /// `keep: {TAG: {has: V}}`; repeatable.
    #[arg(long, num_args = 2, value_names = ["TAG", "V"])]
    has: Vec<String>,
    /// `weigh: {by: [E]}`: a block of one entry (`{when: P, else: w}`);
    /// repeatable.
    #[arg(long, value_name = "E")]
    weigh: Vec<String>,
    /// `weigh: B`: a whole block (`{by: [..], floor: f}`); repeatable.
    #[arg(long, value_name = "B")]
    weigh_block: Vec<String>,
    /// `keep: {$weight: {ge: W}}`; repeatable.
    #[arg(long, value_name = "W")]
    cutoff: Vec<String>,
    /// A sort key, ascending; consecutive sort flags join one `order`.
    #[arg(long, value_name = "K")]
    asc: Vec<String>,
    /// A sort key, descending.
    #[arg(long, value_name = "K")]
    desc: Vec<String>,
    /// `limit: N`; repeatable.
    #[arg(long, value_name = "N")]
    limit: Vec<String>,
    /// `use: NAME`; repeatable.
    #[arg(long = "use", value_name = "NAME")]
    uses: Vec<String>,
    /// `unset: LABEL`: removes the earlier stage labeled LABEL; repeatable.
    ///
    /// LABEL is a stage's `as`, such as a default's.
    #[arg(long, value_name = "LABEL")]
    unset: Vec<String>,
    /// Starts a `per` stage: the flags up to its `--end` are its `run`.
    ///
    /// The group is a facet (`family`) or a list of names
    /// (`'[qwen3, whisper]'`).
    #[arg(long, value_name = "GROUP")]
    per: Vec<String>,
    /// Ends the innermost `--per`.
    #[arg(long, num_args = 0, default_missing_value = "end")]
    end: Vec<String>,
    /// A list of stages, spliced in: a file, or YAML flow; repeatable.
    #[arg(long, value_name = "FILE|[..]")]
    pipeline: Vec<String>,
    #[command(flatten)]
    presets: PresetArgs,
}

/// What [`SelectionStageArgs`] said, read in command-line order: the
/// request's stages, and the presets to apply, which the prefix filters by.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SelectionFlags {
    /// The stages, in order, each attributed to its flag.
    pub stages: Vec<OriginStage>,
    /// The presets: each `--preset`, and each `--on`/`--off` member.
    pub presets: PresetSelection,
    /// The preset flags as written, for the prefix's origin.
    pub preset_flags: Vec<String>,
}

impl SelectionFlags {
    /// The flags of a command flattening [`SelectionStageArgs`], read from
    /// its `matches` in command-line order; `--on X` and `--off X` are read
    /// against `stack`'s declaration of the bool axis `X`.
    ///
    /// # Errors
    /// [`Policy`](bunsen::errors::BunsenErrorKind::Policy) for a flag
    /// whose value doesn't read, an `--on`/`--off` of no bool axis, or a
    /// `--per` without its `--end`.
    pub fn read(
        stack: &PretrainedCatalogStack,
        matches: &ArgMatches,
    ) -> BunsenResult<Self> {
        read_selection(stack, matches).map_err(BunsenError::policy)
    }

    /// These flags with the preset `preset` too, as the flag `flag` wrote
    /// it: a tool's own preset flag (`--think`).
    pub fn with_preset(
        mut self,
        preset: impl Into<String>,
        flag: impl Into<String>,
    ) -> Self {
        self.presets.presets.push(preset.into());
        self.preset_flags.push(flag.into());
        self
    }

    /// The prefix the presets make: one `keep: {$presets: [..]}`, or `None`
    /// without presets.
    pub fn prefix(&self) -> Option<OriginStage> {
        preset_prefix(&self.presets, self.preset_flags.clone())
    }
}

/// One flag of `select`, by its place on the command line.
enum SelectFlag {
    /// A stage, with the flag as written.
    Stage(Stage, String),
    /// A sort key, with the flag as written; consecutive ones join.
    Sort(SortKey, String),
    /// Stages spliced in, with the flag as written.
    Pipeline(Vec<LabeledStage>, String),
    /// `--per GROUP`, with the flag as written.
    PerStart(PerGroup, String),
    /// `--end`.
    PerEnd,
    /// A preset, with the flag as written.
    Preset(String, String),
}

/// The values of the flag `id`, `per` to an occurrence, each with its
/// place on the command line.
fn occurrences(
    matches: &ArgMatches,
    id: &str,
    per: usize,
) -> Vec<(usize, Vec<String>)> {
    let (Some(indices), Some(values)) = (matches.indices_of(id), matches.get_many::<String>(id))
    else {
        return Vec::new();
    };
    let indices: Vec<usize> = indices.collect();
    let values: Vec<String> = values.cloned().collect();
    indices
        .chunks(per)
        .zip(values.chunks(per))
        .map(|(i, v)| (i[0], v.to_vec()))
        .collect()
}

/// A YAML particle, or a file holding one.
fn file_or_flow(text: &str) -> Result<Value, String> {
    let path = std::path::Path::new(text);
    if path.is_file() {
        let body = std::fs::read_to_string(path).map_err(|e| format!("`{text}`: {e}"))?;
        return parse_value(&body);
    }
    parse_value(text)
}

fn stages_of(text: &str) -> Result<Vec<LabeledStage>, String> {
    serde_json::from_value(file_or_flow(text)?)
        .map_err(|e| format!("`{text}` is not a list of stages: {e}"))
}

/// The `select` flags, in command-line order.
fn select_flags(
    stack: &PretrainedCatalogStack,
    matches: &ArgMatches,
) -> Result<Vec<SelectFlag>, String> {
    let mut flags: Vec<(usize, SelectFlag)> = Vec::new();
    let predicate = |text: &str| -> Result<Predicate, String> {
        Predicate::from_value(&parse_value(text)?).map_err(|e| e.problem_message())
    };
    for (id, keep) in [("keep", true), ("drop", false)] {
        for (at, v) in occurrences(matches, id, 1) {
            let p = predicate(&v[0])?;
            let stage = if keep { Stage::Keep(p) } else { Stage::Drop(p) };
            flags.push((at, SelectFlag::Stage(stage, format!("--{id} {}", v[0]))));
        }
    }
    for op in ["lt", "le", "gt", "ge", "eq", "ne", "has"] {
        for (at, v) in occurrences(matches, op, 2) {
            let test = Predicate::Test {
                key: v[0].clone(),
                value: json!({ op: parse_value(&v[1])? }),
            };
            let flag = format!("--{op} {} {}", v[0], v[1]);
            flags.push((at, SelectFlag::Stage(Stage::Keep(test), flag)));
        }
    }
    for (at, v) in occurrences(matches, "weigh", 1) {
        let entry: WeighEntry = serde_json::from_value(parse_value(&v[0])?)
            .map_err(|e| format!("`--weigh {}`: {e}", v[0]))?;
        let block = WeighBlock {
            by: vec![entry],
            floor: None,
        };
        flags.push((
            at,
            SelectFlag::Stage(Stage::Weigh(block), format!("--weigh {}", v[0])),
        ));
    }
    for (at, v) in occurrences(matches, "weigh_block", 1) {
        let block: WeighBlock = serde_json::from_value(parse_value(&v[0])?)
            .map_err(|e| format!("`--weigh-block {}`: {e}", v[0]))?;
        let flag = format!("--weigh-block {}", v[0]);
        flags.push((at, SelectFlag::Stage(Stage::Weigh(block), flag)));
    }
    for (at, v) in occurrences(matches, "cutoff", 1) {
        let w: f64 = v[0]
            .parse()
            .map_err(|_| format!("`--cutoff {}`: not a weight", v[0]))?;
        let test = Predicate::Test {
            key: "$weight".into(),
            value: json!({ "ge": w }),
        };
        flags.push((
            at,
            SelectFlag::Stage(Stage::Keep(test), format!("--cutoff {}", v[0])),
        ));
    }
    for (id, descending) in [("asc", false), ("desc", true)] {
        for (at, v) in occurrences(matches, id, 1) {
            let key = if descending {
                SortKey::desc(v[0].clone())
            } else {
                SortKey::asc(v[0].clone())
            };
            flags.push((at, SelectFlag::Sort(key, format!("--{id} {}", v[0]))));
        }
    }
    for (at, v) in occurrences(matches, "limit", 1) {
        let n: usize = v[0]
            .parse()
            .map_err(|_| format!("`--limit {}`: not a count", v[0]))?;
        flags.push((
            at,
            SelectFlag::Stage(Stage::Limit(n), format!("--limit {}", v[0])),
        ));
    }
    for (at, v) in occurrences(matches, "unset", 1) {
        let flag = format!("--unset {}", v[0]);
        flags.push((at, SelectFlag::Stage(Stage::Unset(v[0].clone()), flag)));
    }
    for (at, v) in occurrences(matches, "per", 1) {
        let group: PerGroup = serde_json::from_value(parse_value(&v[0])?)
            .map_err(|e| format!("`--per {}`: a facet or a list of names: {e}", v[0]))?;
        flags.push((at, SelectFlag::PerStart(group, format!("--per {}", v[0]))));
    }
    for (at, _) in occurrences(matches, "end", 1) {
        flags.push((at, SelectFlag::PerEnd));
    }
    for (at, v) in occurrences(matches, "uses", 1) {
        flags.push((
            at,
            SelectFlag::Stage(Stage::Use(v[0].clone()), format!("--use {}", v[0])),
        ));
    }
    for (at, v) in occurrences(matches, "pipeline", 1) {
        flags.push((
            at,
            SelectFlag::Pipeline(stages_of(&v[0])?, format!("--pipeline {}", v[0])),
        ));
    }
    for (at, v) in occurrences(matches, "presets", 1) {
        flags.push((
            at,
            SelectFlag::Preset(v[0].clone(), format!("--preset {}", v[0])),
        ));
    }
    for (id, on) in [("on", true), ("off", false)] {
        for (at, v) in occurrences(matches, id, 1) {
            let flag = format!("--{id} {}", v[0]);
            let member = bool_axis_member(stack, &v[0], on)
                .map_err(|e| format!("`{flag}`: {}", e.problem_message()))?;
            flags.push((at, SelectFlag::Preset(member, flag)));
        }
    }
    flags.sort_by_key(|(at, _)| *at);
    Ok(flags.into_iter().map(|(_, f)| f).collect())
}

/// [`SelectionFlags::read`], with its errors as messages.
fn read_selection(
    stack: &PretrainedCatalogStack,
    matches: &ArgMatches,
) -> Result<SelectionFlags, String> {
    let mut presets = PresetSelection::default();
    let mut preset_flags = Vec::new();
    // The stages being built: the request's, then one frame per open
    // `--per`, each with its pending sort keys.
    struct Frame {
        stages: Vec<OriginStage>,
        sort: Option<(Vec<SortKey>, Vec<String>)>,
        per: Option<(PerGroup, String)>,
    }
    impl Frame {
        fn new(per: Option<(PerGroup, String)>) -> Self {
            Self {
                stages: Vec::new(),
                sort: None,
                per,
            }
        }

        fn flush(&mut self) {
            if let Some((keys, flags)) = self.sort.take() {
                self.stages.push(OriginStage::new(
                    Stage::Order(SortKeys(keys)),
                    StageOrigin::Flag(flags.join(" ")),
                ));
            }
        }
    }
    let mut frames = vec![Frame::new(None)];
    for flag in select_flags(stack, matches)? {
        let frame = frames.last_mut().expect("the request's frame");
        match flag {
            SelectFlag::Sort(key, text) => {
                let (keys, flags) = frame.sort.get_or_insert_with(Default::default);
                keys.push(key);
                flags.push(text);
            }
            SelectFlag::Preset(preset, text) => {
                presets.presets.push(preset);
                preset_flags.push(text);
            }
            SelectFlag::Stage(stage, text) => {
                frame.flush();
                frame
                    .stages
                    .push(OriginStage::new(stage, StageOrigin::Flag(text)));
            }
            SelectFlag::Pipeline(stages, text) => {
                frame.flush();
                for stage in stages {
                    frame
                        .stages
                        .push(OriginStage::labeled(stage, StageOrigin::Flag(text.clone())));
                }
            }
            SelectFlag::PerStart(group, text) => {
                frame.flush();
                frames.push(Frame::new(Some((group, text))));
            }
            SelectFlag::PerEnd => {
                frame.flush();
                if frames.len() == 1 {
                    return Err("`--end` closes no `--per`".to_string());
                }
                let done = frames.pop().expect("an open `--per`");
                let (group, text) = done.per.expect("a `--per` frame");
                let per = Stage::Per(PerStage {
                    group,
                    run: done.stages.into_iter().map(|s| s.stage).collect(),
                });
                frames
                    .last_mut()
                    .expect("the request's frame")
                    .stages
                    .push(OriginStage::new(
                        per,
                        StageOrigin::Flag(format!("{text} .. --end")),
                    ));
            }
        }
    }
    if frames.len() > 1 {
        let (_, text) = frames.pop().and_then(|f| f.per).expect("a `--per` frame");
        return Err(format!("`{text}` has no `--end`"));
    }
    let mut top = frames.pop().expect("the request's frame");
    top.flush();
    Ok(SelectionFlags {
        stages: top.stages,
        presets,
        preset_flags,
    })
}

/// A build in a result, as `select` prints it: the resolved build's
/// fields, tags as plain values, with its weight.
#[derive(Clone, Debug, PartialEq, Serialize)]
struct ChoiceRecord {
    /// The build's full name.
    name: String,
    /// Its weight.
    weight: f64,
    /// What it is, for a person.
    description: String,
    /// The checkpoint's license.
    #[serde(skip_serializing_if = "Option::is_none")]
    license: Option<String>,
    /// Where the checkpoint comes from.
    #[serde(skip_serializing_if = "Option::is_none")]
    origin: Option<String>,
    /// The template-class chain, root first.
    template_chain: Vec<String>,
    /// Every group it is in.
    memberships: Vec<String>,
    /// Its group in each facet, by facet.
    facets: std::collections::BTreeMap<String, String>,
    /// Its contract, by its name in `$contracts`.
    contract: Option<String>,
    /// Its release membership.
    #[serde(skip_serializing_if = "Option::is_none")]
    release: Option<ReleaseRecord>,
    /// Its tags, written and implied, by full name.
    tags: std::collections::BTreeMap<String, Value>,
    /// The tags a derive step added, by key.
    #[serde(skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    derived: std::collections::BTreeMap<String, Value>,
    /// Its contract's resource slots: each slot's resource.
    resources: std::collections::BTreeMap<String, String>,
    /// Its contract's argument slots: each slot's value.
    #[serde(skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    args: std::collections::BTreeMap<String, Value>,
}

/// A build's release membership, in a [`ChoiceRecord`].
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
struct ReleaseRecord {
    /// The release group.
    group: String,
    /// When the build was released.
    released: String,
}

/// A row a stage removed, as `select --rank` prints it.
#[derive(Clone, Debug, PartialEq, Serialize)]
struct DroppedRecord {
    /// The build's full name.
    name: String,
    /// The stage that removed it.
    stage: usize,
    /// Its weight then.
    weight: f64,
    /// Why.
    why: String,
}

/// What `select` prints.
#[derive(Clone, Debug, PartialEq, Serialize)]
struct SelectionReport {
    /// The composed query.
    query: SelectQuery,
    /// Each stage: where it came from, and how many rows it removed.
    why: Vec<StageTrace>,
    /// With `--rank`, every row a stage removed.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    dropped: Vec<DroppedRecord>,
    /// The presets applied to every result.
    presets: Vec<String>,
    /// The result, in order.
    results: Vec<ResultRecord>,
}

/// One row of a result: a build resolved, or another kind's entry.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(untagged)]
enum ResultRecord {
    /// A build, resolved.
    Build(Box<ChoiceRecord>),
    /// An entry of another kind.
    Entry(EntryRecord),
}

/// An entry of a kind other than `build`, in a result.
#[derive(Clone, Debug, PartialEq, Serialize)]
struct EntryRecord {
    /// Its full name.
    name: String,
    /// Its kind.
    kind: &'static str,
    /// Its weight.
    weight: f64,
}

fn record(
    env: &SelectEnv<'_>,
    name: &CatalogName,
    weight: f64,
) -> Result<ChoiceRecord, String> {
    let entry = env.build(name).map_err(|e| e.problem_message())?;
    let build = &entry.build;
    let names = |names: &mut dyn Iterator<Item = &CatalogName>| -> Vec<String> {
        names.map(ToString::to_string).collect()
    };
    Ok(ChoiceRecord {
        name: build.name.to_string(),
        weight,
        description: build.description.clone(),
        license: build.license.clone(),
        origin: build.origin.clone(),
        template_chain: names(&mut build.template_chain.iter()),
        memberships: names(&mut build.memberships.iter()),
        facets: build
            .facets
            .iter()
            .map(|(facet, group)| (facet.to_string(), group.to_string()))
            .collect(),
        contract: build.contract.as_ref().and_then(CatalogName::contract_path),
        release: build.release.as_ref().map(|r| ReleaseRecord {
            group: r.group.to_string(),
            released: r.released.to_string(),
        }),
        tags: build
            .tags
            .iter()
            .map(|(name, tag)| (name.to_string(), tag.value.clone()))
            .collect(),
        derived: entry.derived.clone(),
        resources: build
            .resources
            .iter()
            .map(|(slot, bound)| (slot.clone(), bound.resource.to_string()))
            .collect(),
        args: build
            .args
            .iter()
            .map(|(slot, bound)| (slot.clone(), bound.value.clone()))
            .collect(),
    })
}

pub(super) fn select(
    out: &mut String,
    stack: &PretrainedCatalogStack,
    matches: &ArgMatches,
    args: SelectArgs,
) -> Result<(), String> {
    let SelectArgs {
        kind,
        names,
        floor,
        defaults,
        rank,
        ..
    } = args;
    let kind = parse_kind(&kind)?;
    let env = SelectEnv::new(stack);
    let context = SelectionContext {
        defaults: defaults
            .as_deref()
            .map(stages_of)
            .transpose()?
            .unwrap_or_default(),
        named: Vec::new(),
    };
    let mut request = SelectRequest {
        kind,
        names,
        stages: Vec::new(),
        floor: floor
            .map(|f| {
                f.parse()
                    .map_err(|_| format!("`--floor {f}`: not a weight"))
            })
            .transpose()?,
    };
    let flags = read_selection(stack, matches)?;
    request.stages = flags.stages.clone();
    let prefix = flags.prefix();
    let presets = flags.presets;
    let composed = context
        .compose(&env, prefix, &request, false)
        .map_err(|e| e.problem_message())?;
    let run: SelectRun = context
        .run_composed(&env, None, &composed)
        .map_err(|e| e.problem_message())?;
    if run.table.is_empty() {
        return Err(empty_error(&run).problem_message());
    }
    let results = run
        .table
        .rows
        .iter()
        .map(|r| {
            if kind == EntryKind::Build {
                record(&env, &r.entry.path, r.weight).map(|c| ResultRecord::Build(Box::new(c)))
            } else {
                Ok(ResultRecord::Entry(EntryRecord {
                    name: r.entry.path.to_string(),
                    kind: kind.as_str(),
                    weight: r.weight,
                }))
            }
        })
        .collect::<Result<_, String>>()?;
    let dropped = if rank {
        run.trace
            .dropped
            .iter()
            .map(|d| DroppedRecord {
                name: d.entry.path.to_string(),
                stage: d.stage,
                weight: d.weight,
                why: d.why.clone(),
            })
            .collect()
    } else {
        Vec::new()
    };
    let report = SelectionReport {
        query: composed.query,
        why: run.trace.stages,
        dropped,
        presets: presets.presets,
        results,
    };
    let yaml = serde_saphyr::to_string(&report).map_err(|e| e.to_string())?;
    emit!(out, "{}", yaml.trim_end());
    Ok(())
}
