use std::str::FromStr;

use burn::{
    module::ParamId,
    prelude::Shape,
};
use xot::{
    Attributes,
    NameId,
    Node,
};

use crate::{
    errors::{
        BunsenError,
        BunsenResult,
    },
    meta::{
        ParamDesc,
        TensorDesc,
        TensorKindDesc,
        TensorParamDesc,
        dtype_from_str,
    },
    modules::reflection::xml_support::{
        names,
        names::{
            CLASS_ATTR,
            DTYPE_ATTR,
            KIND_ATTR,
            PARAM_ELEM,
            PARAM_ID_ATTR,
            RANK_ATTR,
            SHAPE_ATTR,
        },
    },
    zspace::{
        shape_from_xml_attr,
        shape_to_xml_attr,
    },
};

/// Load a [`TensorParamDesc`] from an xml `<Param/>` node.
pub fn node_to_tensor_param_desc(
    xot: &xot::Xot,
    node: xot::Node,
) -> BunsenResult<TensorParamDesc> {
    let param_id_nid = xot.name(PARAM_ID_ATTR).expect("ParamId attribute missing");
    let dtype_nid = xot
        .name(names::DTYPE_ATTR)
        .expect("DType attribute missing");
    let shape_nid = xot
        .name(names::SHAPE_ATTR)
        .expect("Shape attribute missing");
    let kind_nid = xot.name(names::KIND_ATTR).expect("Kind attribute missing");

    let attrs = xot.attributes(node);

    fn get_attr(
        attrs: &Attributes,
        nid: NameId,
        attr: &str,
    ) -> BunsenResult<String> {
        if let Some(val) = attrs.get(nid) {
            return Ok(val.to_string());
        }
        Err(BunsenError::ResourceNotFound(format!(
            "{}/{attr} attribute missing",
            names::PARAM_ELEM,
        )))
    }

    // TODO: Extract, real errors.
    let param_id: ParamId =
        ParamId::deserialize(&get_attr(&attrs, param_id_nid, names::PARAM_ID_ATTR)?);
    let kind = TensorKindDesc::from_str(&get_attr(&attrs, kind_nid, names::KIND_ATTR)?).unwrap();
    let dtype = dtype_from_str(&get_attr(&attrs, dtype_nid, names::DTYPE_ATTR)?)?;
    let shape: Shape = shape_from_xml_attr(&get_attr(&attrs, shape_nid, names::SHAPE_ATTR)?)?;

    Ok(ParamDesc::new(
        param_id,
        TensorDesc::new(kind, dtype, shape),
    ))
}

/// Build an xml `<Param/>` node from a [`TensorParamDesc`].
pub fn tensor_param_desc_to_node(
    xot: &mut xot::Xot,
    param_desc: &TensorParamDesc,
) -> BunsenResult<Node> {
    let name_nid = xot.add_name(PARAM_ELEM);
    let node = xot.new_element(name_nid);
    tensor_param_desc_to_attributes(xot, node, param_desc)
}

/// Write a [`TensorParamDesc`] to the attributes of an xml node.
pub fn tensor_param_desc_to_attributes(
    xot: &mut xot::Xot,
    node: Node,
    param_desc: &TensorParamDesc,
) -> BunsenResult<Node> {
    let param_id_nid = xot.add_name(PARAM_ID_ATTR);
    let class_nid = xot.add_name(CLASS_ATTR);
    let kind_nid = xot.add_name(KIND_ATTR);
    let dtype_nid = xot.add_name(DTYPE_ATTR);
    let shape_nid = xot.add_name(SHAPE_ATTR);
    let rank_nid = xot.add_name(RANK_ATTR);

    xot.set_attribute(node, param_id_nid, param_desc.param_id().to_string());
    xot.set_attribute(node, class_nid, "tensor");

    xot.set_attribute(node, kind_nid, param_desc.kind().to_string());
    xot.set_attribute(node, dtype_nid, format!("{:?}", param_desc.dtype()));

    xot.set_attribute(node, shape_nid, shape_to_xml_attr(param_desc.shape()));
    xot.set_attribute(
        node,
        rank_nid,
        param_desc.shape().clone().rank().to_string(),
    );

    Ok(node)
}
