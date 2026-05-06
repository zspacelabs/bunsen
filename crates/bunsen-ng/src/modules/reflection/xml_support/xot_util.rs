use xot::{
    Node,
    Xot,
};

pub fn pretty_print_node(
    xot: &Xot,
    node: Node,
) -> Result<(), Box<dyn std::error::Error>> {
    println!(
        "{}",
        xot.serialize_xml_string(
            xot::output::xml::Parameters {
                indentation: Some(Default::default()),
                ..Default::default()
            },
            node
        )?
    );

    Ok(())
}
