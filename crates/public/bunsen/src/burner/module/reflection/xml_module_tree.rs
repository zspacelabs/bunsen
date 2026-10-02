#![allow(unused_imports)]
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
    prelude::Backend,
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
    NameId,
    Node,
    Xot,
};

use crate::{
    burner::{
        descriptors::TensorParamDesc,
        module::reflection::{
            module_visitors::XmlModuleTreeBuilder,
            xml_support::{
                adapt_xee_error,
                names,
                node_to_tensor_param_desc,
            },
        },
    },
    errors::BunsenResult,
};

/// The version of the `XmlModuleTree` format.
pub const XML_MODULE_TREE_VERSION: &str = env!("CARGO_PKG_VERSION");

/// An XML document that mirrors a [`Module`]'s structure, for selecting its
/// parameters with `XPath`.
///
/// Build one with [`XmlModuleTree::build`], then:
/// - read everything with [`param_ids`](Self::param_ids) or
///   [`param_descs`](Self::param_descs);
/// - select with [`select_param_ids`](Self::select_param_ids), or build a query
///   with [`query`](Self::query) / [`select`](Self::select) and finish it with
///   an [`XPathModuleQuery`] terminal call;
/// - print it with [`to_xml`](Self::to_xml) to see what to select.
///
/// The tree is a snapshot of structure and parameter ids; it holds no
/// tensors. The element layout, the naming rule (element = type name,
/// `@name` = field name) and an `XPath` crib are in the
/// [module docs](crate::burner::module::reflection).
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
    /// Builds an [`XmlModuleTree`] for a [`Module`].
    ///
    /// Walks the module once with an [`XmlModuleTreeBuilder`]. The tree
    /// records structure and parameter ids; it does not follow later changes
    /// to the module.
    pub fn build<B: Backend, M: Module<B>>(module: &M) -> Self {
        XmlModuleTreeBuilder::build(module)
    }

    /// Creates a new/empty module tree.
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        let mut docs = Documents::new();
        let xot = docs.xot_mut();
        let mtree_nid = xot.add_name(names::XML_MODULE_TREE_ELEM);
        let root = xot.new_element(mtree_nid);

        let _doc = xot.new_document_with_element(root).unwrap();

        let version_nid = xot.add_name("version");
        xot.set_attribute(root, version_nid, XML_MODULE_TREE_VERSION);

        Self { docs, root }
    }

    /// Serializes the module tree to an XML string.
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

    /// Binds (add/lookup) a local (no namespace) name in the [`xot`] arena.
    ///
    /// # Arguments
    /// * `name` - a string name.
    ///
    /// # Returns
    /// A [`NameId`]
    pub fn bind_local_name(
        &mut self,
        name: &str,
    ) -> NameId {
        let xot = self.xot_mut();
        xot.add_name(name)
    }

    /// Binds a list of local names to [`NameId`]s.
    ///
    /// See [`Self::bind_local_name`].
    pub fn bind_local_names<const N: usize>(
        &mut self,
        names: [&str; N],
    ) -> [NameId; N] {
        names.map(|name| self.bind_local_name(name))
    }

    /// The root [`Node`] of the module tree: the `<XmlModuleTree>` element.
    ///
    /// This is only useful with the XML apis.
    pub fn root(&self) -> Node {
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

    /// The [`xot`] arena that backs the document; shorthand for
    /// `self.docs().xot()`.
    ///
    /// For raw XML work the query API does not cover, such as walking
    /// nodes from [`Self::root`].
    pub fn xot(&self) -> &Xot {
        self.docs.xot()
    }

    /// The mutable [`xot`] arena that backs the document; shorthand for
    /// `self.docs_mut().xot_mut()`.
    ///
    /// [`XmlModuleTreeBuilder`] writes the document through this. Nothing
    /// checks edits made through it: a `<Param>` that loses an attribute
    /// makes [`XPathModuleQuery::to_param_descs`] fail.
    pub fn xot_mut(&mut self) -> &mut Xot {
        self.docs.xot_mut()
    }

    /// Returns the [`ParamId`] of every parameter in the module.
    ///
    /// The order is field declaration order; collect into a set when the
    /// order is not the point.
    ///
    /// # Returns
    /// `Ok(Vec<ParamId>)` on success, `Err(e)` on errors.
    ///
    /// # Examples
    /// ```rust
    /// # use burn::nn::{Linear, LinearConfig};
    /// # use bunsen::burner::module::reflection::XmlModuleTree;
    /// # type B = bunsen::support::testing::CpuBackend;
    /// # let device = bunsen::support::testing::default_device();
    /// let module: Linear<B> = LinearConfig::new(2, 3).init(&device);
    /// let mut mtree = XmlModuleTree::build(&module);
    ///
    /// let ids = mtree.param_ids()?;
    /// assert_eq!(ids, [module.weight.id, module.bias.as_ref().unwrap().id]);
    ///
    /// // Is equivalent to (`to_param_ids` applies `.params()` itself):
    /// assert_eq!(mtree.query().to_param_ids()?, ids);
    /// # Ok::<(), bunsen::errors::BunsenError>(())
    /// ```
    pub fn param_ids(&mut self) -> BunsenResult<Vec<ParamId>> {
        self.query().to_param_ids()
    }

    /// Returns a [`TensorParamDesc`] for every parameter in the module.
    ///
    /// # Returns
    /// `Ok(Vec<TensorParamDesc>)` on success, `Err(e)` on errors.
    ///
    /// # Examples
    /// ```rust
    /// # use burn::nn::{Linear, LinearConfig};
    /// # use bunsen::burner::module::reflection::XmlModuleTree;
    /// # type B = bunsen::support::testing::CpuBackend;
    /// # let device = bunsen::support::testing::default_device();
    /// let module: Linear<B> = LinearConfig::new(2, 3).init(&device);
    /// let mut mtree = XmlModuleTree::build(&module);
    ///
    /// let descs = mtree.param_descs()?;
    /// let ranks: Vec<usize> = descs.iter().map(|d| d.rank()).collect();
    /// assert_eq!(ranks, [2, 1]);
    ///
    /// // Is equivalent to (`to_param_descs` applies `.params()` itself):
    /// assert_eq!(mtree.query().to_param_descs()?, descs);
    /// # Ok::<(), bunsen::errors::BunsenError>(())
    /// ```
    pub fn param_descs(&mut self) -> BunsenResult<Vec<TensorParamDesc>> {
        self.query().to_param_descs()
    }

    /// Creates a new default [`XPathModuleQuery`] for this module tree.
    ///
    /// The query builder has a fluent api to incrementally refine a query.
    /// It begins with broad selection over the entire module structure.
    pub fn query<'a>(&'a mut self) -> XPathModuleQuery<'a> {
        XPathModuleQuery::new(
            self,
            format!("/{}/{}", names::XML_MODULE_TREE_ELEM, names::STRUCTURE_ELEM),
        )
    }

    /// Starts a query at `/XmlModuleTree/Structure` and selects `expr` from
    /// there.
    ///
    /// # Panics
    /// On invalid `XPath` expressions.
    ///
    /// # Examples
    /// ```rust
    /// # use burn::nn::{Linear, LinearConfig};
    /// # use bunsen::burner::module::reflection::XmlModuleTree;
    /// # type B = bunsen::support::testing::CpuBackend;
    /// # let device = bunsen::support::testing::default_device();
    /// let module: Linear<B> = LinearConfig::new(2, 3).init(&device);
    /// let mut mtree = XmlModuleTree::build(&module);
    ///
    /// let expected = "/XmlModuleTree/Structure/Linear/*[@name='weight']";
    /// assert_eq!(mtree.select("Linear/*[@name='weight']").expr(), expected);
    ///
    /// // Is equivalent to:
    /// assert_eq!(
    ///     mtree.query().select("Linear/*[@name='weight']").expr(),
    ///     expected
    /// );
    /// ```
    pub fn select<'a>(
        &'a mut self,
        expr: &str,
    ) -> XPathModuleQuery<'a> {
        self.query().select(expr)
    }

    /// Starts a query at `/XmlModuleTree/Structure` and selects `expr` from
    /// there.
    ///
    /// This checks that the expression parses. Errors that only show up
    /// when it is evaluated, such as a type error in a predicate, come from
    /// the terminal call ([`XPathModuleQuery::to_param_ids`] etc.).
    ///
    /// # Returns
    /// `Ok(query)` on success, `Err(e)` on `XPath` errors.
    ///
    /// # Examples
    /// ```rust
    /// # use burn::nn::{Linear, LinearConfig};
    /// # use bunsen::burner::module::reflection::XmlModuleTree;
    /// # type B = bunsen::support::testing::CpuBackend;
    /// # let device = bunsen::support::testing::default_device();
    /// let module: Linear<B> = LinearConfig::new(2, 3).init(&device);
    /// let mut mtree = XmlModuleTree::build(&module);
    ///
    /// let query = mtree.try_select("Linear/*[@name='weight']")?;
    /// assert_eq!(
    ///     query.expr(),
    ///     "/XmlModuleTree/Structure/Linear/*[@name='weight']"
    /// );
    ///
    /// // A syntax error is returned, where `select` would panic:
    /// assert!(mtree.try_select("Linear[").is_err());
    /// # Ok::<(), bunsen::errors::BunsenError>(())
    /// ```
    pub fn try_select<'a>(
        &'a mut self,
        expr: &str,
    ) -> BunsenResult<XPathModuleQuery<'a>> {
        self.query().try_select(expr)
    }

    /// Selects `expr`, then every parameter at or below it.
    ///
    /// # Panics
    /// On invalid `XPath` expressions.
    ///
    /// # Examples
    /// ```rust
    /// # use burn::nn::{Linear, LinearConfig};
    /// # use bunsen::burner::module::reflection::XmlModuleTree;
    /// # type B = bunsen::support::testing::CpuBackend;
    /// # let device = bunsen::support::testing::default_device();
    /// let module: Linear<B> = LinearConfig::new(2, 3).init(&device);
    /// let mut mtree = XmlModuleTree::build(&module);
    ///
    /// let expected = "/XmlModuleTree/Structure/Linear/descendant-or-self::Param";
    /// assert_eq!(mtree.select_params("Linear").expr(), expected);
    ///
    /// // Is equivalent to:
    /// assert_eq!(mtree.query().select("Linear").params().expr(), expected);
    ///
    /// // Is equivalent to:
    /// assert_eq!(
    ///     mtree
    ///         .query()
    ///         .select("Linear/descendant-or-self::Param")
    ///         .expr(),
    ///     expected
    /// );
    /// ```
    pub fn select_params<'a>(
        &'a mut self,
        expr: &str,
    ) -> XPathModuleQuery<'a> {
        self.select(expr).params()
    }

    /// Selects `expr`, then every parameter at or below it.
    ///
    /// The fallible form of [`Self::select_params`]; see
    /// [`Self::try_select`] for which errors it catches.
    ///
    /// # Returns
    /// `Ok(query)` on success, `Err(e)` on `XPath` errors.
    pub fn try_select_params<'a>(
        &'a mut self,
        expr: &str,
    ) -> BunsenResult<XPathModuleQuery<'a>> {
        Ok(self.try_select(expr)?.params())
    }

    /// Returns the [`ParamId`]s of every parameter at or below `expr`.
    ///
    /// The usual way to build an optimizer group's parameter set.
    ///
    /// # Returns
    /// `Ok(Vec<ParamId>)` on success, `Err(e)` on `XPath` errors, including
    /// errors raised while evaluating `expr`.
    ///
    /// # Examples
    /// ```rust
    /// # use burn::nn::{Linear, LinearConfig};
    /// # use bunsen::burner::module::reflection::XmlModuleTree;
    /// # type B = bunsen::support::testing::CpuBackend;
    /// # let device = bunsen::support::testing::default_device();
    /// use std::collections::HashSet;
    ///
    /// use burn::module::ParamId;
    ///
    /// let module: Linear<B> = LinearConfig::new(2, 3).init(&device);
    /// let mut mtree = XmlModuleTree::build(&module);
    ///
    /// let ids: Vec<ParamId> = mtree.select_param_ids("Linear/*[@rank=2]")?;
    /// assert_eq!(ids, [module.weight.id]);
    ///
    /// // Is equivalent to:
    /// assert_eq!(
    ///     mtree.select_params("Linear/*[@rank=2]").to_param_ids()?,
    ///     ids
    /// );
    ///
    /// // Is equivalent to:
    /// assert_eq!(
    ///     mtree
    ///         .query()
    ///         .select("Linear/*[@rank=2]")
    ///         .params()
    ///         .to_param_ids()?,
    ///     ids
    /// );
    ///
    /// // Is equivalent to:
    /// let via_descs: Vec<ParamId> = mtree
    ///     .select_params("Linear/*[@rank=2]")
    ///     .to_param_descs()?
    ///     .iter()
    ///     .map(|d| d.param_id())
    ///     .collect();
    /// assert_eq!(via_descs, ids);
    ///
    /// // Groups are sets; collect into one:
    /// let group: HashSet<ParamId> = ids.into_iter().collect();
    /// # Ok::<(), bunsen::errors::BunsenError>(())
    /// ```
    pub fn select_param_ids(
        &mut self,
        expr: &str,
    ) -> BunsenResult<Vec<ParamId>> {
        self.try_select_params(expr)?.to_param_ids()
    }
}

/// A query builder for an [`XmlModuleTree`].
///
/// This works in two phases:
/// 1. A fluent api to incrementally refine an `XPath` expression.
/// 2. Various execution/output runners to run that expression.
///
/// The builders ([`select`](Self::select), [`filter`](Self::filter), ...)
/// only append text to [`expr`](Self::expr), checking that it parses; the
/// panicking forms panic on a syntax error and the `try_` forms return it.
/// Nothing is evaluated until a terminal call:
/// [`to_param_ids`](Self::to_param_ids),
/// [`to_param_descs`](Self::to_param_descs),
/// [`to_fragments`](Self::to_fragments) or
/// [`xee_execute_many`](Self::xee_execute_many). Evaluation errors come
/// from there.
///
/// A query mutably borrows its tree; finish one query before starting the
/// next. Start one with [`XmlModuleTree::query`] or
/// [`XmlModuleTree::select`].
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

    /// Returns the current `XPath` expression.
    pub fn expr(&self) -> &String {
        &self.expr
    }

    fn try_append_expr(
        self,
        expr: &str,
    ) -> BunsenResult<Self> {
        let expr = format!("{}{}", self.expr, expr);

        Queries::default()
            .many(&expr, |_, _| Ok(()))
            .map_err(|e| adapt_xee_error(e, Some(&expr)))?;

        Ok(Self { expr, ..self })
    }

    /// Executes a [`xee_xpath::Queries::many`] on the current selection.
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

    /// Refines the current selection by appending an `XPath` path expression.
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

    /// Refines the current selection by appending an `XPath` path expression.
    ///
    /// This is: `{EXPR}` => `{EXPR}/{expr}`
    ///
    /// The fallible form of [`Self::select`]. It catches syntax errors only;
    /// evaluation errors come from the terminal call.
    ///
    /// # Returns
    /// `Ok(query)` on success, `Err(e)` on `XPath` errors.
    pub fn try_select<S: AsRef<str>>(
        self,
        expr: S,
    ) -> BunsenResult<XPathModuleQuery<'a>> {
        self.try_append_expr(format!("/{}", expr.as_ref()).as_str())
    }

    /// Refines the current selection by appending an `XPath` predicate
    /// expression.
    ///
    /// A predicate keeps the nodes of the current selection for which it is
    /// true. To require several conditions, call `filter` once per condition
    /// (`[a][b]`) or join them with `and`. A comma is not a conjunction:
    /// `filter("a, b")` fails when evaluated (see the
    /// [crib](crate::burner::module::reflection#xpath-crib)).
    ///
    /// This is: `{EXPR}` => `{EXPR}[{expr}]`
    ///
    /// # Examples
    /// * `filter("@name='foo'")` - select only nodes with a "name" attribute
    ///   equal to "foo".
    /// * `filter("@name='weight'").filter("@rank=2")` - rank-2 nodes named
    ///   "weight".
    ///
    /// # Panics
    /// On invalid `XPath` expressions.
    pub fn filter<S: AsRef<str>>(
        self,
        pred: S,
    ) -> XPathModuleQuery<'a> {
        self.try_filter(pred).unwrap_or_else(|e| panic!("{}", e))
    }

    /// Refines the current selection by appending an `XPath` predicate
    /// expression.
    ///
    /// The fallible form of [`Self::filter`]. It catches syntax errors only;
    /// evaluation errors come from the terminal call.
    ///
    /// This is: `{EXPR}` => `{EXPR}[{expr}]`
    ///
    /// # Returns
    /// `Ok(query)` on success, `Err(e)` on `XPath` errors.
    pub fn try_filter<S: AsRef<str>>(
        self,
        pred: S,
    ) -> BunsenResult<XPathModuleQuery<'a>> {
        self.try_append_expr(format!("[{}]", pred.as_ref()).as_str())
    }

    /// Selects children of the current set.
    ///
    /// This is: "{EXPR}" => "{EXPR}/*"
    pub fn children(self) -> XPathModuleQuery<'a> {
        self.select("*")
    }

    /// Selects children with the given `name` attribute: the struct field
    /// (or enum variant) name, not the type name.
    ///
    /// This is: `{EXPR}` => `{EXPR}/*[@name='{name}']`
    pub fn named_children(
        self,
        name: &str,
    ) -> XPathModuleQuery<'a> {
        self.select(format!("*[@name='{name}']"))
    }

    /// Selects children with the given positional index.
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

    /// Selects every element named `name` (a type name, or `Param`) at or
    /// below the current selection.
    ///
    /// This is: `{EXPR}` => `{EXPR}/descendant-or-self::{name}`
    pub fn subtree_elements(
        self,
        name: &str,
    ) -> XPathModuleQuery<'a> {
        self.select(format!("descendant-or-self::{}", name))
    }

    /// Selects every `<Param>` element at or below the current selection.
    ///
    /// Equivalent to `.subtree_elements("Param")`. The terminal calls
    /// [`Self::to_param_ids`] and [`Self::to_param_descs`] apply it
    /// themselves; call it explicitly to [`filter`](Self::filter) the
    /// parameters, as in `.params().filter("@rank=2")`.
    pub fn params(self) -> Self {
        self.subtree_elements(names::PARAM_ELEM)
    }

    /// Filters the selection to nodes where the `rank` attribute has the given
    /// value.
    ///
    /// Equivalent to `.filter(format!("@rank={rank}"))`
    pub fn rank(
        self,
        rank: usize,
    ) -> Self {
        self.filter(format!("@rank={}", rank))
    }

    /// Returns a [`TensorParamDesc`] for every parameter at or below the
    /// current selection, in document order.
    ///
    /// Implicitly calls [`Self::params`].
    ///
    /// # Returns
    /// `Ok(Vec<TensorParamDesc>)` on success, `Err(e)` on errors, including
    /// errors raised while evaluating the expression.
    pub fn to_param_descs(self) -> BunsenResult<Vec<TensorParamDesc>> {
        let mut query = self.params();

        let nodes: Vec<Node> = query.xee_execute_many(|_, item| Ok(item.to_node()?))?;

        let xot = query.tree.docs.xot();
        nodes
            .into_iter()
            .map(|node| node_to_tensor_param_desc(xot, node))
            .collect::<BunsenResult<Vec<TensorParamDesc>>>()
    }

    /// Returns the [`ParamId`] of every parameter at or below the current
    /// selection, in document order.
    ///
    /// Implicitly calls [`Self::params`].
    ///
    /// # Returns
    /// `Ok(Vec<ParamId>)` on success, `Err(e)` on errors, including errors
    /// raised while evaluating the expression.
    ///
    /// # Examples
    /// ```rust
    /// # use burn::nn::{Linear, LinearConfig};
    /// # use bunsen::burner::module::reflection::XmlModuleTree;
    /// # type B = bunsen::support::testing::CpuBackend;
    /// # let device = bunsen::support::testing::default_device();
    /// use burn::module::ParamId;
    ///
    /// let module: Linear<B> = LinearConfig::new(2, 3).init(&device);
    /// let mut mtree = XmlModuleTree::build(&module);
    ///
    /// let ids = mtree.select("Linear").to_param_ids()?;
    /// assert_eq!(ids, [module.weight.id, module.bias.as_ref().unwrap().id]);
    ///
    /// // Is equivalent to:
    /// let via_descs: Vec<ParamId> = mtree
    ///     .select("Linear")
    ///     .to_param_descs()?
    ///     .iter()
    ///     .map(|d| d.param_id())
    ///     .collect();
    /// assert_eq!(via_descs, ids);
    /// # Ok::<(), bunsen::errors::BunsenError>(())
    /// ```
    pub fn to_param_ids(self) -> BunsenResult<Vec<ParamId>> {
        Ok(self
            .to_param_descs()?
            .iter()
            .map(|d| d.param_id())
            .collect())
    }

    /// Returns the current matches serialized as XML strings; an atomic
    /// result (a string or number) gives its string value.
    ///
    /// For debugging a query: print what it selects.
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
    use crate::{
        burner::descriptors::TensorParamDesc,
        support::testing::{
            CpuBackend,
            default_device,
        },
    };

    #[test]
    fn test_predicate_conjunction() {
        type B = CpuBackend;
        let device = default_device();
        let module: Linear<B> = LinearConfig::new(2, 3).init(&device);
        let mut mtree = XmlModuleTree::build(&module);

        // Stacked predicates and `and` both mean "all of these".
        for expr in [
            "Linear/*[@name='weight'][@rank=2]",
            "Linear/*[@name='weight' and @rank=2]",
        ] {
            assert_eq!(mtree.select_param_ids(expr).unwrap(), [module.weight.id]);
        }
        assert!(
            mtree
                .select_param_ids("Linear/*[@name='bias'][@rank=2]")
                .unwrap()
                .is_empty()
        );

        // A comma builds a two-item sequence, which has no boolean value.
        // The expression parses, so `try_select` accepts it; evaluating it
        // on any node is an XPath type error.
        for expr in [
            "Linear/*[@name='weight',@rank=2]",
            "Linear/*[@name='bias',@rank=2]",
        ] {
            assert!(mtree.try_select(expr).is_ok());
            let err = mtree.select_param_ids(expr).unwrap_err();
            assert!(err.to_string().contains("XPTY0004"), "{err}");
        }

        // Over an empty selection the predicate never runs, so the same
        // mistake hides behind a path that matches nothing.
        assert!(
            mtree
                .select_param_ids("NoSuchType/*[@name='weight',@rank=2]")
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn test_debug() {
        type B = CpuBackend;
        let device = default_device();
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
    fn test_to_xml() {
        type B = CpuBackend;
        let device = default_device();
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
