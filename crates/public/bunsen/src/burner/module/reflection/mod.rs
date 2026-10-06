//! XML/XPath reflection over [`burn::module::Module`]s: select parameters by
//! where they sit in a model.
//!
//! A `burn` module is a Rust type. It tells the compiler about its
//! sub-modules, but it has no structure you can query at run time. You need
//! one whenever you work with a *subset* of a model's parameters, chosen by
//! structure rather than by name in your code:
//!
//! - optimizer groups: the 2-D block weights for Muon, the rest for `AdamW`;
//! - weight decay on some parameters and not others;
//! - audits: which tensors, of which shapes and dtypes, a model holds.
//!
//! Without reflection, each of these is a hand-written
//! [`ModuleVisitor`](burn::module::ModuleVisitor). Here you build an
//! [`XmlModuleTree`] once, an XML document that mirrors the module, and
//! select from it with `XPath` through an [`XPathModuleQuery`]. "Every rank-2
//! weight under the blocks" is one query, and it returns
//! [`ParamId`](burn::module::ParamId)s.
//!
//! The main consumer is [`burner::optim`](crate::burner::optim): the selected
//! `ParamId` sets become its
//! [`OptimizerGroup`](crate::burner::optim::OptimizerGroup)s.
//!
//! This module is behind the `reflection` feature, which is on by default.
//!
//! # The document
//!
//! [`XmlModuleTree::to_xml`] prints the document. For a `Linear` with a bias:
//!
//! ```xml
//! <XmlModuleTree version="...">
//!   <Structure>
//!     <Linear id="n:1" class="struct">
//!       <Param id="n:2" name="weight" param_id="..." class="tensor" kind="Float" dtype="F32" shape="2 3" rank="2"/>
//!       <Param id="n:3" name="bias" param_id="..." class="tensor" kind="Float" dtype="F32" shape="3" rank="1"/>
//!     </Linear>
//!   </Structure>
//! </XmlModuleTree>
//! ```
//!
//! **The element name is the module's *type* name, and the field name is
//! `@name`.** A field `gpt: NanoChatGpt<B>` is `<NanoChatGpt name="gpt">`.
//! Select it with `NanoChatGpt` or `*[@name='gpt']`, never with `gpt`. A path
//! that names a field as an element matches nothing, and an empty selection
//! is not an error. When a query comes back empty, print the dump and read
//! the names off it.
//!
//! - Every query starts at `/XmlModuleTree/Structure`, whose one child is the
//!   root module.
//! - `@class` is `struct` or `enum` for a derived module, `builtin` for the
//!   containers `Vec`, `Array` and `Tuple`, and `tensor` for a `<Param>`.
//! - A struct field has `@name`. The child of an enum module has the variant
//!   name as `@name`: `<Normalization name="norm">` holds `<RmsNorm
//!   name="Rms">`. The items of a `Vec`, `Array` or `Tuple` have no `@name`;
//!   select them by position (`*[2]`, counting from 1).
//! - A `<Param>` is a leaf. Its attributes are those of its
//!   [`TensorParamDesc`](crate::burner::descriptors::TensorParamDesc):
//!   `param_id`, `kind`, `dtype`, `shape` (dims separated by spaces) and
//!   `rank`.
//! - `@id` is unique within one document. It is not stable across builds.
//! - Plain fields (`usize`, `f64`) do not appear. Elements are created as the
//!   visitor enters a module's first field, so a module with no visited fields
//!   at all (`Relu`) has no element of its own. A module whose fields hold no
//!   parameters (plain tensors) is an empty element.
//! - Children are in field declaration order. Prefer sets (`HashSet<ParamId>`)
//!   to relying on that order.
//!
//! # `XPath` crib
//!
//! You don't need all of `XPath`. A step is relative to the current selection,
//! which starts at `/XmlModuleTree/Structure`.
//!
//! | Pattern | Selects |
//! | --- | --- |
//! | `Linear` | Children whose element (type) name is `Linear`. |
//! | `*` | All children, whatever their name. |
//! | `*//Linear` | `Linear` descendants of the children, not the children themselves. At the start, `*` is the root module, so this excludes the root. |
//! | `descendant-or-self::Linear` | All `Linear` elements at or below the current selection; at the start, that includes the root. [`XPathModuleQuery::subtree_elements`] appends this. |
//! | `*[@name='weight']` | Children whose field name is `weight`. |
//! | `*[@rank=2]` | Children whose `rank` attribute is 2. |
//! | `*[2]` | The second child (`XPath` counts from 1). |
//! | `*[@name='weight'][@rank=2]` | Children for which both predicates hold. `*[@name='weight' and @rank=2]` is the same. |
//! | `descendant-or-self::Param` | Every `<Param>` at or below the current selection. [`XPathModuleQuery::params`] appends this. |
//!
//! **`[a, b]` is not "a and b".** In `XPath` a comma builds a sequence, and a
//! sequence of two booleans has no truth value: evaluating
//! `*[@name='weight', @rank=2]` is a type error, `XPTY0004`. The expression
//! parses, so [`XPathModuleQuery::try_select`] accepts it; the error comes
//! from the call that evaluates it ([`XPathModuleQuery::to_param_ids`] and
//! the other terminal calls). Over an empty selection the predicate never
//! runs, so a wrong path in front of it hides the mistake. Stack the
//! predicates instead: `[a][b]`.
//!
//! # Walkthrough
//!
//! An executable tour of the API, one assertion at a time:
//!
//! ```rust
//! use burn::{
//!     module::ParamId,
//!     nn::{
//!         Linear,
//!         LinearConfig,
//!     },
//!     prelude::Backend,
//!     tensor::Shape,
//! };
//!
//! use bunsen::{
//!     errors::{
//!         BunsenError,
//!         BunsenResult,
//!     },
//!     burner::{
//!         descriptors::{
//!             TensorKindDesc,
//!             TensorParamDesc,
//!         },
//!         module::reflection::{
//!             XML_MODULE_TREE_VERSION,
//!             XmlModuleTree,
//!             XPathModuleQuery,
//!         },
//!     },
//! };
//!
//! use bunsen::support::testing::cpu_device;
//! type B = bunsen::support::testing::CpuBackend;
//! let device = cpu_device();
//!
//! // Create a Linear module, with a bias:
//! // * `weight` - `Param<Tensor<B, 2>>` [d_input, d_output].
//! // * `bias` - `Option<Param<Tensor<B, 1>>>` [d_output].
//! let d_input = 2;
//! let d_output = 3;
//! let module: Linear<B> = LinearConfig::new(d_input, d_output).init(&device);
//!
//! // [`TensorParamDesc`] can describe a `Param<Tensor<B, R, K>>`:
//! let weight_desc: TensorParamDesc = TensorParamDesc::from(&module.weight);
//! let bias_ref = module.bias.as_ref().unwrap();
//! let bias_desc: TensorParamDesc = TensorParamDesc::from(bias_ref);
//!
//! // [`TensorParamDesc`] exposes the basic `Param` and `Tensor` metadata:
//! assert_eq!(weight_desc.param_id(), module.weight.id);
//!
//! // [`TensorKindDesc`] is an enum which describes the current kind variants:
//! assert_eq!(weight_desc.kind(), TensorKindDesc::Float);
//! assert_eq!(weight_desc.dtype(), module.weight.dtype());
//!
//! assert_eq!(weight_desc.shape(), &module.weight.shape());
//! assert_eq!(weight_desc.shape(), &Shape::new([d_input, d_output]));
//!
//! // [`TensorParamDesc`] also provides some convenience methods:
//! assert_eq!(weight_desc.rank(), 2);
//! assert_eq!(weight_desc.num_elements(), 2 * 3);
//! assert_eq!(
//!     weight_desc.num_elements(),
//!     weight_desc.shape().num_elements()
//! );
//!
//! // This is a rough size-estimate of the buffer size used by the parameter.
//! assert_eq!(
//!     weight_desc.size_estimate(),
//!     module.weight.dtype().size() * 2 * 3
//! );
//!
//! // Build an XmlModuleTree from the module.
//! // Evaluating XPath needs mutable access to the document,
//! // so the tree must be `mut` to be queried.
//! let mut mtree = XmlModuleTree::build(&module);
//!
//! // [`XmlModuleTree::to_xml`] dumps the document; see "The document" above.
//! //
//! // The structure sits inside wrapping elements to leave room for other
//! // metadata later. Every query starts at `/XmlModuleTree/Structure`.
//! //
//! // The element name comes from the `container_type` that `burn`'s
//! // [`burn::module::ModuleVisitor::enter_module`] reports:
//! // * `{TYPE}` => NAME=TYPE, CLASS='builtin'
//! // * `{C}:{TYPE}` => NAME=TYPE, CLASS=lowercase(C)
//! // `Param` elements are always class "tensor".
//! assert_eq!(
//!     mtree.to_xml(true),
//!     indoc::formatdoc! {r#"
//!             <XmlModuleTree version="{XML_MODULE_TREE_VERSION}">
//!               <Structure>
//!                 <Linear id="n:1" class="struct">
//!                   <Param id="n:2" name="weight" param_id="{weight_id}" class="tensor" kind="Float" dtype="{weight_dtype}" shape="2 3" rank="2"/>
//!                   <Param id="n:3" name="bias" param_id="{bias_id}" class="tensor" kind="Float" dtype="{bias_dtype}" shape="3" rank="1"/>
//!                 </Linear>
//!               </Structure>
//!             </XmlModuleTree>
//!             "#,
//!             weight_id = weight_desc.param_id(),
//!             weight_dtype = format!("{:?}", weight_desc.dtype()),
//!             bias_id = bias_desc.param_id(),
//!             bias_dtype = format!("{:?}", bias_desc.dtype()),
//!         }
//! );
//!
//! // [`XmlModuleTree`] has a Debug impl:
//! assert_eq!(
//!     format!("{:#?}", mtree),
//!     indoc::formatdoc! {r#"
//!             XmlModuleTree {{
//!               <XmlModuleTree version="{XML_MODULE_TREE_VERSION}">
//!                 <Structure>
//!                   <Linear id="n:1" class="struct">
//!                     <Param id="n:2" name="weight" param_id="{weight_id}" class="tensor" kind="Float" dtype="{weight_dtype}" shape="2 3" rank="2"/>
//!                     <Param id="n:3" name="bias" param_id="{bias_id}" class="tensor" kind="Float" dtype="{bias_dtype}" shape="3" rank="1"/>
//!                   </Linear>
//!                 </Structure>
//!               </XmlModuleTree>
//!             }}"#,
//!             weight_id = weight_desc.param_id(),
//!             weight_dtype = format!("{:?}", weight_desc.dtype()),
//!             bias_id = bias_desc.param_id(),
//!             bias_dtype = format!("{:?}", bias_desc.dtype()),
//!         }
//! );
//!
//! // [`XmlModuleTree::param_ids`] returns all [`ParamId`]s, as a `Vec`.
//! //
//! // It is shorthand for `mtree.query().to_param_ids()`, checked below.
//! // `to_param_ids` applies `.params()` itself, which is
//! // `.select("descendant-or-self::Param")`.
//! let module_param_ids: Vec<ParamId> = mtree.param_ids()?;
//!
//! // IMPORTANT: Module Tree Ordering
//! //
//! // `burner` Modules order their children in a stable and specific order,
//! // determined by the order of their declaration in the source code,
//! // and the current semantics of the `Module` derive macro.
//! //
//! // Where possible, you should not rely upon this; and should prefer
//! // to use `HashSet<ParamId>` or similar to shield yourself from
//! // ordering variation; particularly as you'll generally be using
//! // this machinery when doing subset calculations.
//! assert_eq!(
//!     module_param_ids,
//!     [module.weight.id, module.bias.as_ref().unwrap().id]
//! );
//!
//! // [`XPathModuleQuery::to_param_ids`] returns the [`ParamId`]s of every
//! // parameter in the selected subtrees.
//! assert_eq!(
//!     &mtree.query().to_param_ids()?,
//!     &module_param_ids,
//! );
//!
//! // [`XmlModuleTree::param_descs`] returns a description of every parameter.
//! //
//! // This leverages the [`TensorParamDesc`] API to strip generics from
//! // the introspection api.
//! //
//! // Like [`XmlModuleTree::param_ids`], it is shorthand for
//! // `mtree.query().to_param_descs()`, checked below.
//! let module_param_descs: Vec<TensorParamDesc> = mtree.param_descs()?;
//! assert_eq!(
//!     &module_param_descs,
//!     &vec![weight_desc.clone(), bias_desc.clone()]
//! );
//!
//! // [`XPathModuleQuery::to_param_descs`] returns a [`TensorParamDesc`]
//! // for every parameter in the selected subtrees.
//! assert_eq!(
//!         &mtree
//!             .query()
//!             .to_param_descs()?,
//!         &module_param_descs,
//!     );
//!
//! // The query api is designed to be fluent and chainable.
//! //
//! // The [`XPathModuleQuery<'a>`] captures a borrow of the module tree,
//! // so you'll need to resolve the borrow before running another query.
//! let mut query: XPathModuleQuery<'_> = mtree.query();
//!
//! // We can introspect on the current XPath expression being accumulated
//! // by a query by calling `expr()`.
//! assert_eq!(query.expr(), "/XmlModuleTree/Structure");
//!
//! // [`XPathModuleQuery`] has a Debug impl:
//! assert_eq!(
//!     format!("{:#?}", query),
//!     indoc::formatdoc! {r#"
//!         XPathModuleQuery {{
//!             tree: XmlModuleTree {{
//!               <XmlModuleTree version="{version}">
//!                 <Structure>
//!                   <Linear id="n:1" class="struct">
//!                     <Param id="n:2" name="weight" param_id="{weight_id}" class="tensor" kind="Float" dtype="{weight_dtype}" shape="2 3" rank="2"/>
//!                     <Param id="n:3" name="bias" param_id="{bias_id}" class="tensor" kind="Float" dtype="{bias_dtype}" shape="3" rank="1"/>
//!                   </Linear>
//!                 </Structure>
//!               </XmlModuleTree>
//!             }},
//!             expr: "/XmlModuleTree/Structure",
//!         }}"#,
//!         version=XML_MODULE_TREE_VERSION,
//!         weight_id = weight_desc.param_id(),
//!         weight_dtype = format!("{:?}", weight_desc.dtype()),
//!             bias_id = bias_desc.param_id(),
//!             bias_dtype = format!("{:?}", bias_desc.dtype()),
//!         }
//! );
//!
//! // We can collect the current query results as XML fragments.
//! // This is primarily useful for debugging.
//! //
//! // Initially, this will be the root Module node.
//! assert_eq!(
//!         &query.to_fragments(true)?,
//!         &[indoc::formatdoc! {r#"
//!             <Structure>
//!               <Linear id="n:1" class="struct">
//!                 <Param id="n:2" name="weight" param_id="{weight_id}" class="tensor" kind="Float" dtype="{weight_dtype}" shape="2 3" rank="2"/>
//!                 <Param id="n:3" name="bias" param_id="{bias_id}" class="tensor" kind="Float" dtype="{bias_dtype}" shape="3" rank="1"/>
//!               </Linear>
//!             </Structure>"#,
//!             weight_id = weight_desc.param_id(),
//!             weight_dtype = format!("{:?}", weight_desc.dtype()),
//!             bias_id = bias_desc.param_id(),
//!             bias_dtype = format!("{:?}", bias_desc.dtype()),
//!         },],
//!     );
//!
//! // The [`XPathModuleQuery::params`] method selects all the `Param` elements
//! // in the current subtree.
//! let mut query = mtree.query().params();
//! assert_eq!(
//!     query.expr(),
//!     "/XmlModuleTree/Structure/descendant-or-self::Param"
//! );
//! assert_eq!(
//!     &query.to_fragments(false)?,
//!     &[
//!         format!(
//!             r#"<Param id="n:2" name="weight" param_id="{weight_id}" class="tensor" kind="Float" dtype="{weight_dtype}" shape="2 3" rank="2"/>"#,
//!             weight_id = weight_desc.param_id(),
//!             weight_dtype = format!("{:?}", weight_desc.dtype()),
//!         ),
//!         format!(
//!             r#"<Param id="n:3" name="bias" param_id="{bias_id}" class="tensor" kind="Float" dtype="{bias_dtype}" shape="3" rank="1"/>"#,
//!             bias_id = bias_desc.param_id(),
//!             bias_dtype = format!("{:?}", bias_desc.dtype()),
//!         )
//!     ],
//! );
//!
//! // A full coverage of the XPath language cannot be included here.
//! // For more details, see: <https://en.wikipedia.org/wiki/XPath>
//!
//! // The structural elements start at '/XmlModuleTree/Structure/$Elem'.
//! // There is exactly one root element: the root module.
//! //
//! // We can select this using either:
//! // - the element selector (here, "Linear").
//! // - the wildcard selector ('*').
//! // - (a bunch of other, longer XPath operators).
//! //
//! // "Linear":
//! // - Select the root 'Linear' node,
//! let mut query = mtree.query().select("Linear");
//! assert_eq!(query.expr(), "/XmlModuleTree/Structure/Linear");
//! assert_eq!(
//!     &query.to_fragments(true)?,
//!     &[indoc::formatdoc! {r#"
//!         <Linear id="n:1" class="struct">
//!           <Param id="n:2" name="weight" param_id="{weight_id}" class="tensor" kind="Float" dtype="{weight_dtype}" shape="2 3" rank="2"/>
//!           <Param id="n:3" name="bias" param_id="{bias_id}" class="tensor" kind="Float" dtype="{bias_dtype}" shape="3" rank="1"/>
//!         </Linear>"#,
//!         weight_id = weight_desc.param_id(),
//!         weight_dtype = format!("{:?}", weight_desc.dtype()),
//!         bias_id = bias_desc.param_id(),
//!         bias_dtype = format!("{:?}", bias_desc.dtype()),
//!     },],
//! );
//!
//! // Here's the same thing using the wildcard selector:
//! //
//! // "*":
//! // - Select the root's children, which is only the 'Linear' node
//! let mut query = mtree.query().select("*");
//! assert_eq!(query.expr(), "/XmlModuleTree/Structure/*");
//! assert_eq!(
//!     &query.to_fragments(true)?,
//!     &[indoc::formatdoc! {r#"
//!         <Linear id="n:1" class="struct">
//!           <Param id="n:2" name="weight" param_id="{weight_id}" class="tensor" kind="Float" dtype="{weight_dtype}" shape="2 3" rank="2"/>
//!           <Param id="n:3" name="bias" param_id="{bias_id}" class="tensor" kind="Float" dtype="{bias_dtype}" shape="3" rank="1"/>
//!         </Linear>"#,
//!         weight_id = weight_desc.param_id(),
//!         weight_dtype = format!("{:?}", weight_desc.dtype()),
//!         bias_id = bias_desc.param_id(),
//!         bias_dtype = format!("{:?}", bias_desc.dtype()),
//!     },],
//! );
//!
//! // "*//Linear" selects descendants of the children, not the children
//! // themselves, so here it misses the root module.
//! // "descendant-or-self::Linear" (`subtree_elements`) includes it:
//! assert!(mtree.select("*//Linear").to_fragments(false)?.is_empty());
//! assert_eq!(
//!     mtree.query().subtree_elements("Linear").to_param_ids()?,
//!     module_param_ids,
//! );
//!
//! // We can select specific names of structural elements by name,
//! // using an attribute predicated `[@name='name']`.
//! //
//! // "Linear/*[@name='weight']":
//! // - Select the root 'Linear' node,
//! // - Select all the children of 'Linear',
//! // - Filter those to elements with the attribute 'name' set to 'weight'.
//! //
//! // "expr[predicate]" keeps the nodes of `expr` for which the predicate
//! // is true.
//! let mut query = mtree.query().select("Linear/*[@name='weight']");
//! assert_eq!(
//!     query.expr(),
//!     "/XmlModuleTree/Structure/Linear/*[@name='weight']"
//! );
//! assert_eq!(
//!     &query.to_fragments(false)?,
//!     &[format!(
//!         r#"<Param id="n:2" name="weight" param_id="{weight_id}" class="tensor" kind="Float" dtype="{weight_dtype}" shape="2 3" rank="2"/>"#,
//!         weight_id = weight_desc.param_id(),
//!         weight_dtype = format!("{:?}", weight_desc.dtype()),
//!     ),],
//! );
//!
//! // Predicates stack: "expr[a][b]" keeps the nodes for which both hold,
//! // and "expr[a and b]" is the same.
//! assert_eq!(
//!     mtree.select_param_ids("Linear/*[@name='weight'][@rank=2]")?,
//!     [module.weight.id],
//! );
//! assert_eq!(
//!     mtree.select_param_ids("Linear/*[@name='weight' and @rank=2]")?,
//!     [module.weight.id],
//! );
//!
//! // A comma is NOT "and". "[a, b]" is a sequence of two booleans, which
//! // has no truth value, so evaluating it is an XPath type error. It
//! // parses, so building the query succeeds; evaluating it fails:
//! let comma = "Linear/*[@name='weight', @rank=2]";
//! assert!(mtree.try_select(comma).is_ok());
//! let err = mtree.select_param_ids(comma).unwrap_err();
//! assert!(err.to_string().contains("XPTY0004"));
//!
//! // We can also select the children of elements by their index.
//! // Note: XPath indexes from 1, not 0.
//! //
//! // The children of sequences in the "builtin" class ('Tuple', 'Vec', 'Array')
//! // don't have names, but do have positional indices in XPath.
//! //
//! // "Linear/*[2]":
//! // - Select the root 'Linear' node,
//! // - Select all the children of 'Linear',
//! // - Select the 2nd (indexing from 1) child.
//! let mut query = mtree.query().select("Linear/*[2]");
//! assert_eq!(query.expr(), "/XmlModuleTree/Structure/Linear/*[2]");
//! assert_eq!(
//!     &query.to_fragments(false)?,
//!     &[format!(
//!         r#"<Param id="n:3" name="bias" param_id="{bias_id}" class="tensor" kind="Float" dtype="{bias_dtype}" shape="3" rank="1"/>"#,
//!         bias_id = bias_desc.param_id(),
//!         bias_dtype = format!("{:?}", bias_desc.dtype()),
//!     )],
//! );
//!
//! // In general:
//! // - `query.select(expr)` appends "/expr" to the current query expression.
//! // - `query.filter(expr)` appends "[expr]" to the current query expression.
//! //
//! // `query.params()` may seem superfluous, as both `.to_param_ids()` and
//! // `.to_param_descs()` implicitly call `.params()`.
//! //
//! // However, when used in conjunction with `.filter()`, we can write powerful
//! // selection expressions.
//! let mut query = mtree.query().params().filter("@rank=2");
//! assert_eq!(
//!     query.expr(),
//!     "/XmlModuleTree/Structure/descendant-or-self::Param[@rank=2]"
//! );
//! assert_eq!(
//!     &query.to_fragments(false)?,
//!     &[format!(
//!         r#"<Param id="n:2" name="weight" param_id="{weight_id}" class="tensor" kind="Float" dtype="{weight_dtype}" shape="2 3" rank="2"/>"#,
//!         weight_id = weight_desc.param_id(),
//!         weight_dtype = format!("{:?}", weight_desc.dtype()),
//!     ),],
//! );
//!
//! // Many structural builtin components are also Modules.
//! let module = (
//!     LinearConfig::new(2, 3).init::<B>(&device),
//!     [LinearConfig::new(4, 5).init::<B>(&device)],
//!     vec![
//!         LinearConfig::new(6, 7).init::<B>(&device),
//!         LinearConfig::new(8, 9).init::<B>(&device),
//!     ],
//! );
//! let expected_dtype = module.0.weight.dtype();
//! let dtype_str = format!("{:?}", expected_dtype);
//! // So we can still walk these module:
//! let mut mtree = XmlModuleTree::build(&module);
//! assert_eq!(
//!         &mtree.query().to_fragments(true)?,
//!         &[indoc::formatdoc! {r#"
//!             <Structure>
//!               <Tuple id="n:1" class="builtin">
//!                 <Linear id="n:2" class="struct">
//!                   <Param id="n:3" name="weight" param_id="{}" class="tensor" kind="Float" dtype="{dtype}" shape="2 3" rank="2"/>
//!                   <Param id="n:4" name="bias" param_id="{}" class="tensor" kind="Float" dtype="{dtype}" shape="3" rank="1"/>
//!                 </Linear>
//!                 <Array id="n:5" class="builtin">
//!                   <Linear id="n:6" class="struct">
//!                     <Param id="n:7" name="weight" param_id="{}" class="tensor" kind="Float" dtype="{dtype}" shape="4 5" rank="2"/>
//!                     <Param id="n:8" name="bias" param_id="{}" class="tensor" kind="Float" dtype="{dtype}" shape="5" rank="1"/>
//!                   </Linear>
//!                 </Array>
//!                 <Vec id="n:9" class="builtin">
//!                   <Linear id="n:A" class="struct">
//!                     <Param id="n:B" name="weight" param_id="{}" class="tensor" kind="Float" dtype="{dtype}" shape="6 7" rank="2"/>
//!                     <Param id="n:C" name="bias" param_id="{}" class="tensor" kind="Float" dtype="{dtype}" shape="7" rank="1"/>
//!                   </Linear>
//!                   <Linear id="n:D" class="struct">
//!                     <Param id="n:E" name="weight" param_id="{}" class="tensor" kind="Float" dtype="{dtype}" shape="8 9" rank="2"/>
//!                     <Param id="n:F" name="bias" param_id="{}" class="tensor" kind="Float" dtype="{dtype}" shape="9" rank="1"/>
//!                   </Linear>
//!                 </Vec>
//!               </Tuple>
//!             </Structure>"#,
//!             module.0.weight.id,
//!             module.0.bias.as_ref().unwrap().id,
//!             module.1[0].weight.id,
//!             module.1[0].bias.as_ref().unwrap().id,
//!             module.2[0].weight.id,
//!             module.2[0].bias.as_ref().unwrap().id,
//!             module.2[1].weight.id,
//!             module.2[1].bias.as_ref().unwrap().id,
//!             dtype=dtype_str,
//!         }],
//!     );
//!
//! // We could select all the `Linear` descendants:
//! let mut query = mtree.select("*//Linear");
//! assert_eq!(query.expr(), "/XmlModuleTree/Structure/*//Linear");
//! assert_eq!(
//!     &query.to_fragments(true)?,
//!     &[
//!         indoc::formatdoc! {r#"
//!           <Linear id="n:2" class="struct">
//!             <Param id="n:3" name="weight" param_id="{}" class="tensor" kind="Float" dtype="{dtype}" shape="2 3" rank="2"/>
//!             <Param id="n:4" name="bias" param_id="{}" class="tensor" kind="Float" dtype="{dtype}" shape="3" rank="1"/>
//!           </Linear>"#,
//!             module.0.weight.id,
//!             module.0.bias.as_ref().unwrap().id,
//!             dtype=dtype_str,
//!         },
//!         indoc::formatdoc! {r#"
//!           <Linear id="n:6" class="struct">
//!             <Param id="n:7" name="weight" param_id="{}" class="tensor" kind="Float" dtype="{dtype}" shape="4 5" rank="2"/>
//!             <Param id="n:8" name="bias" param_id="{}" class="tensor" kind="Float" dtype="{dtype}" shape="5" rank="1"/>
//!           </Linear>"#,
//!             module.1[0].weight.id,
//!             module.1[0].bias.as_ref().unwrap().id,
//!             dtype=dtype_str,
//!         },
//!         indoc::formatdoc! {r#"
//!           <Linear id="n:A" class="struct">
//!             <Param id="n:B" name="weight" param_id="{}" class="tensor" kind="Float" dtype="{dtype}" shape="6 7" rank="2"/>
//!             <Param id="n:C" name="bias" param_id="{}" class="tensor" kind="Float" dtype="{dtype}" shape="7" rank="1"/>
//!           </Linear>"#,
//!             module.2[0].weight.id,
//!             module.2[0].bias.as_ref().unwrap().id,
//!             dtype=dtype_str,
//!         },
//!         indoc::formatdoc! {r#"
//!           <Linear id="n:D" class="struct">
//!             <Param id="n:E" name="weight" param_id="{}" class="tensor" kind="Float" dtype="{dtype}" shape="8 9" rank="2"/>
//!             <Param id="n:F" name="bias" param_id="{}" class="tensor" kind="Float" dtype="{dtype}" shape="9" rank="1"/>
//!           </Linear>"#,
//!             module.2[1].weight.id,
//!             module.2[1].bias.as_ref().unwrap().id,
//!             dtype=dtype_str,
//!         },
//!     ],
//! );
//!
//! // Returning to the power of .params().filter("@rank=2"), we can see the effect
//! // on complex module:
//! let mut query = mtree.query().params().filter("@rank=2");
//! assert_eq!(
//!     query.expr(),
//!     "/XmlModuleTree/Structure/descendant-or-self::Param[@rank=2]"
//! );
//! assert_eq!(
//!         &query.to_fragments(false)?,
//!         &[
//!             format!(
//!                 r#"<Param id="n:3" name="weight" param_id="{}" class="tensor" kind="Float" dtype="{dtype}" shape="2 3" rank="2"/>"#,
//!                 module.0.weight.id,
//!                 dtype = dtype_str,
//!             ),
//!             format!(
//!                 r#"<Param id="n:7" name="weight" param_id="{}" class="tensor" kind="Float" dtype="{dtype}" shape="4 5" rank="2"/>"#,
//!                 module.1[0].weight.id,
//!                 dtype = dtype_str,
//!             ),
//!             format!(
//!                 r#"<Param id="n:B" name="weight" param_id="{}" class="tensor" kind="Float" dtype="{dtype}" shape="6 7" rank="2"/>"#,
//!                 module.2[0].weight.id,
//!                 dtype = dtype_str,
//!             ),
//!             format!(
//!                 r#"<Param id="n:E" name="weight" param_id="{}" class="tensor" kind="Float" dtype="{dtype}" shape="8 9" rank="2"/>"#,
//!                 module.2[1].weight.id,
//!                 dtype = dtype_str,
//!             ),
//!         ],
//!     );
//!
//! Ok::<(), BunsenError>(())
//! ```

pub mod module_visitors;
pub mod xml_support;

mod xml_module_tree;
pub use xml_module_tree::*;
