#![allow(unused)]

use std::{
    fmt::{
        Debug,
        Display,
    },
    str::FromStr,
};

use burn::{
    module::{
        Module,
        ModuleVisitor,
        ParamId,
    },
    nn::{
        Linear,
        LinearConfig,
    },
    prelude::{
        Backend,
        Shape,
    },
};
use xee_xpath::{
    Documents,
    Itemable,
    Queries,
    Query,
    error::{
        ErrorValue,
        Result as SpannedResult,
    },
    query::Convert,
};
use xot::{
    Attribute,
    Attributes,
    NameId,
    Node,
    Xot,
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
    modules::reflection::{
        module_visitors::XmlModuleTreeBuilder,
        xml_support::{
            adapt_xee_error,
            names,
            node_to_tensor_param_desc,
        },
    },
    zspace::shape_from_xml_attr,
};

pub const XML_MODULE_TREE_VERSION: &str = env!("CARGO_PKG_VERSION");

/// XML/XPath reflection layer for burn [`Module`]s.
pub struct XmlModuleTree {
    docs: Documents,
    root: Node,
}

impl Debug for XmlModuleTree {
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        f.write_str("XmlModuleTree {")?;
        if f.alternate() {
            f.write_str("\n")?;
            for line in self.to_xml(true).lines() {
                writeln!(f, "  {line}")?;
            }
        } else {
            f.write_str(self.to_xml(false).as_str())?;
        }
        f.write_str("}")
    }
}

impl XmlModuleTree {
    /// Build a [`XmlModuleTree`] for a [`Module`].
    pub fn build<B: Backend, M: Module<B>>(module: &M) -> Self {
        XmlModuleTreeBuilder::build(module)
    }

    /// Create a new/empty module tree.
    pub(crate) fn new() -> Self {
        let mut docs = Documents::new();
        let xot = docs.xot_mut();
        let mtree_nid = xot.add_name(names::XML_MODULE_TREE_ELEM);
        let root = xot.new_element(mtree_nid);
        let doc = xot.new_document_with_element(root).unwrap();

        let version_nid = xot.add_name("version");
        xot.set_attribute(root, version_nid, XML_MODULE_TREE_VERSION);

        Self { docs, root }
    }

    /// Serialize the module tree to an XML string.
    pub fn to_xml(
        &self,
        pretty: bool,
    ) -> String {
        self.docs
            .xot()
            .serialize_xml_string(
                xot::output::xml::Parameters {
                    indentation: if pretty {
                        Some(Default::default())
                    } else {
                        None
                    },
                    ..Default::default()
                },
                self.root,
            )
            .unwrap()
    }

    /// Bind (add/lookup) a local (no namespace) name in the [`xot`] arena.
    ///
    /// # Arguments
    /// * `name` - a string name.
    ///
    /// # Returns
    /// A [`NameId`]
    pub(crate) fn bind_local_name(
        &mut self,
        name: &str,
    ) -> NameId {
        let xot = self.xot_mut();
        xot.add_name(name)
    }

    /// Bind a list of local names to [`NameId`]s.
    ///
    /// See [`bind_local_name`].
    pub(crate) fn bind_local_names<const N: usize>(
        &mut self,
        names: [&str; N],
    ) -> [NameId; N] {
        names.map(|name| self.bind_local_name(name))
    }

    /// The root [`Node`] document node of the module tree.
    ///
    /// This is only useful with the XML apis.
    pub(crate) fn root(&self) -> Node {
        self.root
    }

    /// A const view of the [`xee_xpath`] [`Documents`] arena.
    pub fn docs(&self) -> &Documents {
        &self.docs
    }

    /// A mut view of the [`xee_xpath`] [`Documents`] arena.
    pub fn docs_mut(&mut self) -> &mut Documents {
        &mut self.docs
    }

    /// Internal. Shorthand access to the [`xot`] arena.
    pub(crate) fn xot(&self) -> &Xot {
        self.docs.xot()
    }

    /// Internal. Shorthand access to the mutable [`xot`] arena.
    pub(crate) fn xot_mut(&mut self) -> &mut Xot {
        self.docs.xot_mut()
    }

    /// Iterate over [`ParamId`]s for each parameter in the subtree.
    ///
    /// Implicitly calls [`XPathModuleQuery::params`].
    ///
    /// # Returns
    /// `Ok(Vec<ParamId>)` on success, `Err(e)` on errors.
    ///
    /// ## Example
    /// ```rust,ignore
    /// let param_ids: HashSet<ParamId> = mtree
    ///     .param_ids()?
    ///     .collect();
    ///
    /// // Is equivalent to:
    /// let param_ids: HashSet<ParamId> = mtree
    ///     .query()
    ///     // .params() is implicit to [`ModuleTreeQuery::to_param_ids`],
    ///     // equivalent to: .select("descendant-or-self::Param")
    ///     .to_param_ids()?
    ///     .collect();
    /// ```
    pub fn param_ids(&mut self) -> BunsenResult<Vec<ParamId>> {
        self.query().to_param_ids()
    }

    /// Iterate over [`TensorParamDesc`]s for each parameter in the subtree.
    ///
    /// Implicitly calls [`XPathModuleQuery::params`].
    ///
    /// # Returns
    /// `Ok(Vec<TensorParamDesc>)` on success, `Err(e)` on errors.
    ///
    /// ## Example
    /// ```rust,ignore
    /// let descs: Vec<ParamDesc<TensorDesc>> = mtree
    ///     .param_descs()?
    ///     .collect();
    ///
    /// // Is equivalent to:
    /// let descs: Vec<ParamDesc<TensorDesc>> = mtree
    ///     .query()
    ///     // .params() is implicit to [`ModuleTreeQuery::to_param_descs`],
    ///     // equivalent to: .select("descendant-or-self::Param")
    ///     .to_param_descs()?
    ///     .collect();
    /// ```
    pub fn param_descs(&mut self) -> BunsenResult<Vec<TensorParamDesc>> {
        self.query().to_param_descs()
    }

    /// Create a new default [`XPathModuleQuery`] for this module tree.
    ///
    /// The query builder has a fluent api to incrementally refine a query.
    /// It begins with broad selection over the entire module structure.
    pub fn query<'a>(&'a mut self) -> XPathModuleQuery<'a> {
        XPathModuleQuery::new(
            self,
            format!("/{}/{}", names::XML_MODULE_TREE_ELEM, names::STRUCTURE_ELEM),
        )
    }

    /// Query a sub-tree.
    ///
    /// # Panics
    /// On invalid `XPath` expressions.
    ///
    /// # Example
    /// ```rust,ignore
    /// let q: QueryBuilder<_> = mtree
    ///     .select("GPT/Linear");
    ///
    /// // Is equivalent to:
    /// let q: QueryBuilder<_> = mtree
    ///     .query()
    ///     .select("GPT/Linear");
    /// ```
    pub fn select<'a>(
        &'a mut self,
        expr: &str,
    ) -> XPathModuleQuery<'a> {
        self.query().select(expr)
    }

    /// Query a sub-tree.
    ///
    /// # Returns
    /// `Ok(query)` on success, `Err(e)` on `XPath` errors.
    ///
    /// # Example
    /// ```rust,ignore
    /// let q: QueryBuilder<_> = mtree
    ///     .select("GPT/Linear");
    ///
    /// // Is equivalent to:
    /// let q: QueryBuilder<_> = mtree
    ///     .query()
    ///     .select("GPT/Linear");
    /// ```
    pub fn try_select<'a>(
        &'a mut self,
        expr: &str,
    ) -> BunsenResult<XPathModuleQuery<'a>> {
        self.query().try_select(expr)
    }

    /// Query all parameters of a subtree.
    ///
    /// # Panics
    /// On invalid `XPath` expressions.
    ///
    /// # Example
    /// ```rust,ignore
    /// let q: QueryBuilder<_> = mtree
    ///     .select_params("GPT/Linear");
    ///
    /// // Is equivalent to:
    /// let q: QueryBuilder<_> = mtree
    ///     .query()
    ///     .select("GPT/Linear")
    ///     .params();
    ///
    /// // Is equivalent to:
    /// let q: QueryBuilder<_> = mtree
    ///     .query()
    ///     .select("GPT/Linear/descedant-or-self::Param");
    /// ```
    pub fn select_params<'a>(
        &'a mut self,
        expr: &str,
    ) -> XPathModuleQuery<'a> {
        self.select(expr).params()
    }

    /// Query all parameters of a subtree.
    ///
    /// # Returns
    /// `Ok(query)` on success, `Err(e)` on `XPath` errors.
    ///
    /// # Example
    /// ```rust,ignore
    /// let q: QueryBuilder<_> = mtree
    ///     .select_params("GPT/Linear");
    ///
    /// // Is equivalent to:
    /// let q: QueryBuilder<_> = mtree
    ///     .query()
    ///     .select("GPT/Linear")
    ///     .params();
    ///
    /// // Is equivalent to:
    /// let q: QueryBuilder<_> = mtree
    ///     .query()
    ///     .select("GPT/Linear/descedant-or-self::Param");
    /// ```
    pub fn try_select_params<'a>(
        &'a mut self,
        expr: &str,
    ) -> BunsenResult<XPathModuleQuery<'a>> {
        Ok(self.try_select(expr)?.params())
    }

    /// Return an iterator over the parameter [`ParamId`]s of a subtree.
    ///
    /// # Returns
    /// `Ok(impl Iterator<Item = ParamId>)` on success, `Err(e)` on `XPath`
    /// errors.
    ///
    /// # Example
    /// ```rust,ignore
    /// let param_ids : HashSet<ParamId> = mtree
    ///     .select_param_ids("GPT/Linear")?
    ///     .collect();
    ///
    /// # Is equivalent to:
    /// let param_ids : HashSet<ParamId> = mtree
    ///     .select_params("GPT/Linear")
    ///     .to_param_ids()?
    ///     .collect();
    ///
    /// # Is equivalent to:
    /// let param_ids : HashSet<ParamId> = mtree
    ///     .query()
    ///     .select("GPT/Linear")
    ///     .params()
    ///     .to_param_ids()?
    ///     .collect();
    ///
    /// # Is equivalent to:
    /// let param_ids : HashSet<ParamId> = mtree
    ///     .query()
    ///     .select("GPT/Linear/descendant-or-self::Param")
    ///     .to_param_ids()?
    ///     .collect();
    ///
    /// # Is equivalent to:
    /// let param_ids : HashSet<ParamId> = mtree
    ///     .query()
    ///     .select("GPT/Linear/descendant-or-self::Param")
    ///     .to_param_descs()?
    ///     .map(|d| d.param_id())
    ///     .collect();
    /// ```
    pub fn select_param_ids(
        &mut self,
        expr: &str,
    ) -> BunsenResult<Vec<ParamId>> {
        self.try_select_params(expr)?.to_param_ids()
    }
}

/// A query builder for a [`XmlModuleTree`].
///
/// This works in two phases:
/// 1. A fluent api to incrementally refine an `XPath` expression.
/// 2. Various execution/output runners to run that expression.
#[derive(Debug)]
pub struct XPathModuleQuery<'a> {
    tree: &'a mut XmlModuleTree,
    expr: String,
}

impl<'a> XPathModuleQuery<'a> {
    /// Create a new [`XPathModuleQuery`].
    ///
    /// See: [`XmlModuleTree::query`].
    fn new(
        tree: &'a mut XmlModuleTree,
        expr: String,
    ) -> Self {
        Self { tree, expr }
    }

    /// Get the current `XPath` expression.
    pub fn expr(&self) -> &String {
        &self.expr
    }

    fn try_append_expr(
        mut self,
        expr: &str,
    ) -> BunsenResult<Self> {
        let expr = format!("{}{}", self.expr, expr);

        Queries::default()
            .many(&expr, |_, _| Ok(()))
            .map_err(|e| adapt_xee_error(e, Some(&expr)))?;

        Ok(Self { expr, ..self })
    }

    /// Execute a [`xee_xpath::Queries::many`] on the current selection.
    ///
    /// This exposes the `xot`/`xee_xpath` query execution functionality.
    pub fn xee_execute_many<V, F>(
        &mut self,
        f: F,
    ) -> BunsenResult<Vec<V>>
    where
        F: Convert<V>,
    {
        let expr = &self.expr;
        let root = self.tree.root;

        Queries::default()
            .many(expr, f)
            .map_err(|e| adapt_xee_error(e, Some(expr)))?
            .execute(&mut self.tree.docs, root)
            .map_err(|e| adapt_xee_error(e, Some(expr)))
    }

    /// Refine the current selection by appending an `XPath` path expression.
    ///
    /// This is: `{EXPR}` => `{EXPR}/{expr}`
    ///
    /// # Panics
    /// On invalid `XPath` expressions.
    pub fn select<S: AsRef<str>>(
        self,
        expr: S,
    ) -> XPathModuleQuery<'a> {
        self.try_select(expr).unwrap_or_else(|e| panic!("{}", e))
    }

    /// Refine the current selection by appending an `XPath` path expression.
    ///
    /// This is: `{EXPR}` => `{EXPR}/{expr}`
    ///
    /// # Returns
    /// `Ok(query)` on success, `Err(e)` on `XPath` errors.
    pub fn try_select<S: AsRef<str>>(
        self,
        expr: S,
    ) -> BunsenResult<XPathModuleQuery<'a>> {
        self.try_append_expr(format!("/{}", expr.as_ref()).as_str())
    }

    /// Refine the current selection by appending an `XPath` predicate
    /// expression.
    ///
    /// Predicate expressions filter the current node-set, keeping only those
    /// nodes for which each branch of the predicate expression evaluates to
    /// true.
    ///
    /// This is: `{EXPR}` => `{EXPR}[{expr}]`
    ///
    /// # Example
    /// * `filter("@name='foo'")` - select only nodes with a "name" attribute
    ///   equal to "foo".
    ///
    /// # Panics
    /// On invalid `XPath` expressions.
    pub fn filter<S: AsRef<str>>(
        self,
        pred: S,
    ) -> XPathModuleQuery<'a> {
        self.try_filter(pred).unwrap_or_else(|e| panic!("{}", e))
    }

    /// Refine the current selection by appending an `XPath` predicate
    /// expression.
    ///
    /// Predicate expressions filter the current node-set, keeping only those
    /// nodes for which each branch of the predicate expression evaluates to
    /// true.
    ///
    /// This is: `{EXPR}` => `{EXPR}[{expr}]`
    ///
    /// # Example
    /// * `filter("@name='foo'")` - select only nodes with a "name" attribute
    ///   equal to "foo".
    ///
    /// # Returns
    /// `Ok(query)` on success, `Err(e)` on `XPath` errors.
    pub fn try_filter<S: AsRef<str>>(
        self,
        pred: S,
    ) -> BunsenResult<XPathModuleQuery<'a>> {
        self.try_append_expr(format!("[{}]", pred.as_ref()).as_str())
    }

    /// Select children of the current set.
    ///
    /// This is: "{EXPR}" => "{EXPR}/*"
    pub fn children(self) -> XPathModuleQuery<'a> {
        self.select("*")
    }

    /// Select children withh the given `name` attribute.
    ///
    /// This is: `{EXPR}` => `{EXPR}/*[@name='{name}']`
    pub fn named_children(
        self,
        name: &str,
    ) -> XPathModuleQuery<'a> {
        self.select(format!("*[@name='{name}']"))
    }

    /// Select children with the given positional index.
    ///
    /// NOTE: `XPath` indexing is 1-based; so this method adds 1 to the index.
    ///
    /// This is: `{EXPR}` => `{EXPR}/*[{index + 1}]`
    pub fn indexed_children(
        self,
        index: usize,
    ) -> XPathModuleQuery<'a> {
        self.select(format!("*[{}]", index + 1))
    }

    /// Recursively select all descedant or self elements with `name`.
    ///
    /// This is: `{EXPR}` => `{EXPR}/descendant-or-self::{name}`
    pub fn subtree_elements(
        self,
        name: &str,
    ) -> XPathModuleQuery<'a> {
        self.select(format!("descendant-or-self::{}", name))
    }

    /// Recursively select all parameter elements in the current context.
    ///
    /// Equivalent to `.desdendant_or_self_elem(names::PARAM_ELEM)`
    pub fn params(self) -> Self {
        self.subtree_elements(names::PARAM_ELEM)
    }

    /// Filter the selection to nodes where the `rank` attribute has the given
    /// value.
    ///
    /// Equivalent to `.filter(format!("@rank={rank}"))`
    pub fn rank(
        self,
        rank: usize,
    ) -> Self {
        self.filter(format!("@rank={}", rank))
    }

    /// Iterate over [`TensorParamDesc`]s for each parameter in the subtree.
    ///
    /// Implicitly calls [`Self::params`].
    ///
    /// # Returns
    /// `Ok(Vec<TensorParamDesc>)` on success, `Err(e)` on
    /// errors.
    pub fn to_param_descs(mut self) -> BunsenResult<Vec<TensorParamDesc>> {
        let mut query = self.params();

        let nodes: Vec<Node> = query.xee_execute_many(|docs, item| Ok(item.to_node()?))?;

        let xot = query.tree.docs.xot();
        nodes
            .into_iter()
            .map(|node| node_to_tensor_param_desc(xot, node))
            .collect::<BunsenResult<Vec<ParamDesc<TensorDesc>>>>()
    }

    /// Iterate over [`ParamId`]s for each parameter in the subtree.
    ///
    /// Implicitly calls [`Self::params`].
    ///
    /// # Returns
    /// `Ok(Vec<ParamId>)` on success, `Err(e)` on errors.
    ///
    /// # Example
    /// ```rust,ignore
    /// let param_ids : HashSet<ParamId> = query
    ///     .to_param_ids()?
    ///     .collect();
    ///
    /// # Is equivalent to:
    /// let param_ids : HashSet<ParamId> = query
    ///     .to_param_descs()?
    ///     .map(|d| d.param_id())
    ///     .collect();
    /// ```
    pub fn to_param_ids(mut self) -> BunsenResult<Vec<ParamId>> {
        Ok(self
            .to_param_descs()?
            .iter()
            .map(|d| d.param_id())
            .collect())
    }

    /// Iterate over string fragments for the current expression matches.
    ///
    /// # Arguments
    /// * `pretty` - pretty-print/indent the xml fragments.
    pub fn to_fragments(
        &mut self,
        pretty: bool,
    ) -> BunsenResult<Vec<String>> {
        use xee_xpath::Item;

        let output_params = xot::output::xml::Parameters {
            indentation: if pretty {
                Some(Default::default())
            } else {
                None
            },
            ..Default::default()
        };

        let fragments = self.xee_execute_many(
            |docs: &mut Documents, item: &Item| -> SpannedResult<String> {
                let xot: &xot::Xot = docs.xot();

                match item {
                    Item::Node(node) => xot
                        .serialize_xml_string(output_params.clone(), *node)
                        .map(|s| s.trim().to_string())
                        .map_err(|e| ErrorValue::from(e).into()),
                    _ => Ok(item.string_value(xot)?),
                }
            },
        )?;

        Ok(fragments)
    }
}

#[cfg(test)]
#[allow(unused)]
mod tests {
    use burn::nn::{
        Linear,
        LinearConfig,
    };

    use super::*;

    #[test]
    #[cfg(feature = "cuda")]
    fn test_debug_cuda() {
        test_debug::<burn::backend::Cuda>();
    }

    #[test]
    #[cfg(feature = "wgpu")]
    fn test_debug_wgpu() {
        test_debug::<burn::backend::Wgpu>();
    }

    fn test_debug<B: Backend>() {
        let device = Default::default();
        let module: Linear<B> = LinearConfig::new(2, 3).init(&device);

        let weight_desc: TensorParamDesc = TensorParamDesc::from(&module.weight);
        let bias_ref = module.bias.as_ref().unwrap();
        let bias_desc: TensorParamDesc = TensorParamDesc::from(bias_ref);

        let mut mtree = XmlModuleTree::build(&module);

        assert_eq!(
            format!("{:#?}", mtree),
            indoc::formatdoc! {r#"
                XmlModuleTree {{
                  <XmlModuleTree version="{version}">
                    <Structure>
                      <Linear id="n:1" class="struct">
                        <Param id="n:2" name="weight" param_id="{weight_id}" class="tensor" kind="Float" dtype="{weight_dtype}" shape="2 3" rank="2"/>
                        <Param id="n:3" name="bias" param_id="{bias_id}" class="tensor" kind="Float" dtype="{bias_dtype}" shape="3" rank="1"/>
                      </Linear>
                    </Structure>
                  </XmlModuleTree>
                }}"#,
                version=XML_MODULE_TREE_VERSION,
                weight_id = weight_desc.param_id(),
                weight_dtype = format!("{:?}", weight_desc.dtype()),
                bias_id = bias_desc.param_id(),
                bias_dtype = format!("{:?}", bias_desc.dtype()),
            }
        );

        assert_eq!(
            format!("{:?}", mtree),
            indoc::formatdoc! {r#"XmlModuleTree {{<XmlModuleTree version="{XML_MODULE_TREE_VERSION}"><Structure><Linear id="n:1" class="struct"><Param id="n:2" name="weight" param_id="{weight_id}" class="tensor" kind="Float" dtype="{weight_dtype}" shape="2 3" rank="2"/><Param id="n:3" name="bias" param_id="{bias_id}" class="tensor" kind="Float" dtype="{bias_dtype}" shape="3" rank="1"/></Linear></Structure></XmlModuleTree>}}"#,
                weight_id = weight_desc.param_id(),
                weight_dtype = format!("{:?}", weight_desc.dtype()),
                bias_id = bias_desc.param_id(),
                bias_dtype = format!("{:?}", bias_desc.dtype()),
            }
        );
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_to_xml_cuda() {
        test_to_xml::<burn::backend::Cuda>();
    }

    #[test]
    #[cfg(feature = "wgpu")]
    fn test_to_xml_wgpu() {
        test_to_xml::<burn::backend::Wgpu>();
    }

    fn test_to_xml<B: Backend>() {
        let device = Default::default();
        let module: Linear<B> = LinearConfig::new(2, 3).init(&device);

        let weight_desc: TensorParamDesc = TensorParamDesc::from(&module.weight);
        let bias_ref = module.bias.as_ref().unwrap();
        let bias_desc: TensorParamDesc = TensorParamDesc::from(bias_ref);

        let mut mtree = XmlModuleTree::build(&module);

        assert_eq!(
            mtree.to_xml(true),
            indoc::formatdoc! {r#"
                <XmlModuleTree version="{version}">
                  <Structure>
                    <Linear id="n:1" class="struct">
                      <Param id="n:2" name="weight" param_id="{weight_id}" class="tensor" kind="Float" dtype="{weight_dtype}" shape="2 3" rank="2"/>
                      <Param id="n:3" name="bias" param_id="{bias_id}" class="tensor" kind="Float" dtype="{bias_dtype}" shape="3" rank="1"/>
                    </Linear>
                  </Structure>
                </XmlModuleTree>
                "#,
                version=XML_MODULE_TREE_VERSION,
                weight_id = weight_desc.param_id(),
                weight_dtype = format!("{:?}", weight_desc.dtype()),
                bias_id = bias_desc.param_id(),
                bias_dtype = format!("{:?}", bias_desc.dtype()),
            }
        );

        assert_eq!(
            mtree.to_xml(false),
            indoc::formatdoc! {r#"<XmlModuleTree version="{version}"><Structure><Linear id="n:1" class="struct"><Param id="n:2" name="weight" param_id="{weight_id}" class="tensor" kind="Float" dtype="{weight_dtype}" shape="2 3" rank="2"/><Param id="n:3" name="bias" param_id="{bias_id}" class="tensor" kind="Float" dtype="{bias_dtype}" shape="3" rank="1"/></Linear></Structure></XmlModuleTree>"#,
                version=XML_MODULE_TREE_VERSION,
                weight_id = weight_desc.param_id(),
                weight_dtype = format!("{:?}", weight_desc.dtype()),
                bias_id = bias_desc.param_id(),
                bias_dtype = format!("{:?}", bias_desc.dtype()),
            }
        );
    }
}
