use std::{
    fmt::Display,
    str::FromStr,
};

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
    burner::{
        descriptors::{
            ParamDesc,
            TensorDesc,
            TensorKindDesc,
            TensorParamDesc,
            dtype_from_str,
        },
        module::reflection::xml_support::{
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
            shape_from_xml_attr,
            shape_to_xml_attr,
        },
    },
    errors::{
        BunsenError,
        BunsenErrorKind,
        BunsenResult,
        LookupError,
        ParseError,
        ResultContext,
    },
};

/// Loads a [`TensorParamDesc`] from an xml `<Param/>` node.
///
/// # Errors
/// - [`Lookup`](crate::errors::BunsenErrorKind::Lookup), with a [`LookupError`]
///   cause, if the node lacks an attribute;
/// - [`Internal`](crate::errors::BunsenErrorKind::Internal) if its `param_id`,
///   `kind`, `dtype` or `shape` attribute does not parse, or `kind` disagrees
///   with `dtype`: module reflection writes them itself.
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
        Err(BunsenError::lookup(LookupError::missing("attribute", attr))
            .context(format!("<{}>", names::PARAM_ELEM)))
    }

    fn parse_attr<T>(
        attrs: &Attributes,
        nid: NameId,
        attr: &str,
    ) -> BunsenResult<T>
    where
        T: FromStr,
        T::Err: Display,
    {
        let val = get_attr(attrs, nid, attr)?;
        val.parse().map_err(|e: T::Err| {
            BunsenError::from_cause(
                BunsenErrorKind::Internal,
                ParseError::new(format!("<{}> {attr} attribute", names::PARAM_ELEM))
                    .input(&val)
                    .because(e),
            )
        })
    }

    let param_id: ParamId = parse_attr(&attrs, param_id_nid, names::PARAM_ID_ATTR)?;
    let kind: TensorKindDesc = parse_attr(&attrs, kind_nid, names::KIND_ATTR)?;
    let dtype = dtype_from_str(&get_attr(&attrs, dtype_nid, names::DTYPE_ATTR)?)
        .as_internal()
        .with_context(|| format!("<{}> {} attribute", names::PARAM_ELEM, names::DTYPE_ATTR))?;
    if kind != TensorKindDesc::from(dtype) {
        return Err(BunsenError::internal(format!(
            "<{}> {} attribute {kind} does not match {} attribute {dtype:?}",
            names::PARAM_ELEM,
            names::KIND_ATTR,
            names::DTYPE_ATTR,
        )));
    }

    let shape: Shape = shape_from_xml_attr(&get_attr(&attrs, shape_nid, names::SHAPE_ATTR)?)
        .as_internal()
        .with_context(|| format!("<{}> {} attribute", names::PARAM_ELEM, names::SHAPE_ATTR))?;

    Ok(ParamDesc::new(param_id, TensorDesc::new(dtype, shape)))
}

/// Builds an xml `<Param/>` node from a [`TensorParamDesc`].
pub fn tensor_param_desc_to_node(
    xot: &mut xot::Xot,
    param_desc: &TensorParamDesc,
) -> BunsenResult<Node> {
    let name_nid = xot.add_name(PARAM_ELEM);
    let node = xot.new_element(name_nid);
    tensor_param_desc_to_attributes(xot, node, param_desc)
}

/// Writes a [`TensorParamDesc`] to the attributes of an xml node.
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

#[cfg(test)]
mod tests {
    use burn::tensor::DType;

    use super::*;
    use crate::errors::BunsenErrorKind;

    fn param_node(
        xot: &mut xot::Xot,
        param_id: &str,
    ) -> Node {
        let desc = ParamDesc::new(
            ParamId::new(),
            TensorDesc::new(DType::F32, Shape::new([2, 3])),
        );
        let node = tensor_param_desc_to_node(xot, &desc).unwrap();
        let param_id_nid = xot.add_name(PARAM_ID_ATTR);
        xot.set_attribute(node, param_id_nid, param_id);
        node
    }

    #[test]
    fn test_round_trip() {
        let mut xot = xot::Xot::new();
        let desc = ParamDesc::new(
            ParamId::new(),
            TensorDesc::new(DType::F32, Shape::new([2, 3])),
        );
        let node = tensor_param_desc_to_node(&mut xot, &desc).unwrap();
        assert_eq!(node_to_tensor_param_desc(&xot, node).unwrap(), desc);
    }

    #[test]
    fn test_bad_param_id_is_internal() {
        let mut xot = xot::Xot::new();
        let node = param_node(&mut xot, "not a param id!");
        let err = node_to_tensor_param_desc(&xot, node).unwrap_err();
        assert_eq!(err.kind(), BunsenErrorKind::Internal);
        assert!(err.to_string().contains("param_id"), "{err}");
    }
}
