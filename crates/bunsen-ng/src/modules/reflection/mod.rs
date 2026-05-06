//! XML/XPath reflection layer for burn [`burn::module::Module`]s.
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
//! use bunsen_ng::{
//!     errors::BunsenResult,
//!     modules::reflection::{
//!         XML_MODULE_TREE_VERSION,
//!         XmlModuleTree,
//!         XPathModuleQuery,
//!     },
//!     meta::{
//!         TensorKindDesc,
//!         TensorParamDesc,
//!     },
//! };
//!
//! #[cfg(feature = "cuda")]
//! basic_module_tree_api_example::<burn::backend::Cuda>().unwrap();
//!
//! fn basic_module_tree_api_example<B: Backend>() -> BunsenResult<()> {
//!     let device = Default::default();
//!
//!     // Create a Linear module, with a bias:
//!     // * `weight` - `Param<Tensor<B, 2>>` [d_input, d_output].
//!     // * `bias` - `Option<Param<Tensor<B, 1>>>` [d_output].
//!     let d_input = 2;
//!     let d_output = 3;
//!     let module: Linear<B> = LinearConfig::new(d_input, d_output).init(&device);
//!
//!     // [`TensorParamDesc`] can describe a `Param<Tensor<B, R, K>>`:
//!     let weight_desc: TensorParamDesc = TensorParamDesc::from(&module.weight);
//!     let bias_ref = module.bias.as_ref().unwrap();
//!     let bias_desc: TensorParamDesc = TensorParamDesc::from(bias_ref);
//!
//!     // [`TensorParamDesc`] exposes the basic `Param` and `Tensor` metadata:
//!     assert_eq!(weight_desc.param_id(), module.weight.id);
//!
//!     // [`TensorKindDesc`] is an enum which describes the current kind variants:
//!     assert_eq!(weight_desc.kind(), TensorKindDesc::Float);
//!     assert_eq!(weight_desc.dtype(), module.weight.dtype());
//!
//!     assert_eq!(weight_desc.shape(), &module.weight.shape());
//!     assert_eq!(weight_desc.shape(), &Shape::new([d_input, d_output]));
//!
//!     // [`TensorParamDesc`] also provides some convience methods:
//!     assert_eq!(weight_desc.rank(), 2);
//!     assert_eq!(weight_desc.num_elements(), 2 * 3);
//!     assert_eq!(
//!         weight_desc.num_elements(),
//!         weight_desc.shape().num_elements()
//!     );
//!
//!     // This is a rough size-estimate of the buffer size used by the parameter.
//!     assert_eq!(
//!         weight_desc.size_estimate(),
//!         module.weight.dtype().size() * 2 * 3
//!     );
//!
//!     // Build a XmlModuleTree from the module.
//!     // As the XmlModuleTree holds non-Send active active query environment,
//!     // it must be `mut` to be useful.
//!     let mut mtree = XmlModuleTree::build(&module);
//!
//!     // [`XmlModuleTree`] builds an XML meta-description of the module structure.
//!     //
//!     // This can be dumped directly to a `String` to examine the module structure.
//!     //
//!     // The XPath expressions used by query api all all written in terms of this
//!     // structure; though they start with `/Module/Structure` as their implied
//!     // context.
//!     //
//!     // The module structure is embedded in the wrapping elements to provide
//!     // a pathway to future metadata extension.
//!     //
//!     // # @id - Document-Unique Id
//!     // Every structural element has a document-unique id attribute, which can be
//!     // used to reference the element in the XML.
//!     //
//!     // # <{NAME} class="{CLASS}"/> - Structural Element
//!     // Structural elements are given a {NAME} and {CLASS} in the local namespace,
//!     // derived from the [`burn::module::ModuleVisitor::enter_module`]
//!     // `container_type`.
//!     // * `{TYPE}` => NAME=TYPE, CLASS='builtin'
//!     // * `{C}:{TYPE}` => NAME=TYPE, CLASS=lowercase(C)
//!     //
//!     // # @class - Element Class
//!     // Structural elements derive their class from their `container_type`;
//!     // while `Param` elements are (currently) always "tensor".
//!     //
//!     // # @name - The structural field name.
//!     // If an element is a named field of a "struct"-class parent,
//!     // then it will have a `@name` attribute.
//!     assert_eq!(
//!         mtree.to_xml(true),
//!         indoc::formatdoc! {r#"
//!                 <XmlModuleTree version="{XML_MODULE_TREE_VERSION}">
//!                   <Structure>
//!                     <Linear id="n:1" class="struct">
//!                       <Param id="n:2" name="weight" param_id="{weight_id}" class="tensor" kind="Float" dtype="{weight_dtype}" shape="2 3" rank="2"/>
//!                       <Param id="n:3" name="bias" param_id="{bias_id}" class="tensor" kind="Float" dtype="{bias_dtype}" shape="3" rank="1"/>
//!                     </Linear>
//!                   </Structure>
//!                 </XmlModuleTree>
//!                 "#,
//!                 weight_id = weight_desc.param_id(),
//!                 weight_dtype = format!("{:?}", weight_desc.dtype()),
//!                 bias_id = bias_desc.param_id(),
//!                 bias_dtype = format!("{:?}", bias_desc.dtype()),
//!             }
//!     );
//!
//!     // [`XmlModuleTree`] has a Debug impl:
//!     assert_eq!(
//!         format!("{:#?}", mtree),
//!         indoc::formatdoc! {r#"
//!                 XmlModuleTree {{
//!                   <XmlModuleTree version="{XML_MODULE_TREE_VERSION}">
//!                     <Structure>
//!                       <Linear id="n:1" class="struct">
//!                         <Param id="n:2" name="weight" param_id="{weight_id}" class="tensor" kind="Float" dtype="{weight_dtype}" shape="2 3" rank="2"/>
//!                         <Param id="n:3" name="bias" param_id="{bias_id}" class="tensor" kind="Float" dtype="{bias_dtype}" shape="3" rank="1"/>
//!                       </Linear>
//!                     </Structure>
//!                   </XmlModuleTree>
//!                 }}"#,
//!                 weight_id = weight_desc.param_id(),
//!                 weight_dtype = format!("{:?}", weight_desc.dtype()),
//!                 bias_id = bias_desc.param_id(),
//!                 bias_dtype = format!("{:?}", bias_desc.dtype()),
//!             }
//!     );
//!
//!     // [`XmlModuleTree::param_ids`] iterates over all [`ParamId`]s.
//!     //
//!     // This is a useful way to get all the parameter ids in a module;
//!     // but it is actually a wrapper over a series of more complex steps.
//!     //
//!     //   let ids: Vec<ParamId> = mtree
//!     //       .query()
//!     //       // .params() is implicit to [`XPathModuleQuery::to_param_ids`],
//!     //       // equivalent to: .select("descendant-or-self::Param")
//!     //       .to_param_ids()?
//!     //       .collect();
//!     let module_param_ids: Vec<ParamId> = mtree.param_ids()?;
//!
//!     // IMPORTANT: Module Tree Ordering
//!     //
//!     // `burn` Modules order their children in a stable and specific order,
//!     // determined by the order of their declaration in the source code,
//!     // and the current semantics of the `Module` derive macro.
//!     //
//!     // Where possible, you should not rely upon this; and should prefer
//!     // to use `HashSet<ParamId>` or similar to shield yourself from
//!     // ordering variation; particularly as you'll generally be using
//!     // this machinery when doing subset calculations.
//!     assert_eq!(
//!         module_param_ids,
//!         [module.weight.id, module.bias.as_ref().unwrap().id]
//!     );
//!
//!     // [`XPathModuleQuery::to_param_ids`] iterates over [`ParamId`]s for
//!     // each parameter in the subtree.
//!     assert_eq!(
//!             &mtree.query().to_param_ids()?,
//!             &module_param_ids,
//!         );
//!
//!     // [`XmlModuleTree::param_descs`] iterates over descriptions of every parameter.
//!     //
//!     // This leverages the [`TensorParamDesc`] API to strip generics from
//!     // the introspection api.
//!     //
//!     // Similar to [`XmlModuleTree::param_ids`], this is a wrapper over a series of more
//!     // complex steps.
//!     //
//!     //   let descs: Vec<ParamDesc<TensorDesc>> = mtree
//!     //       .query()
//!     //       // .params() is implicit to [`XPathModuleQuery::to_param_descs`],
//!     //       // equivalent to: .select("descendant-or-self::Param")
//!     //       .to_param_descs()?
//!     //       .collect();
//!     let module_param_descs: Vec<TensorParamDesc> = mtree.param_descs()?;
//!     assert_eq!(
//!         &module_param_descs,
//!         &vec![weight_desc.clone(), bias_desc.clone()]
//!     );
//!
//!     // [`XPathModuleQuery::to_param_descs`] iterates over
//!     // [`TensorParamDesc`]s for each parameter in the subtree.
//!     assert_eq!(
//!             &mtree
//!                 .query()
//!                 .to_param_descs()?,
//!             &module_param_descs,
//!         );
//!
//!     // The query api is designed to be fluent and chainable.
//!     //
//!     // The [`XPathModuleQuery<'a>`] captures a borrow of the module tree,
//!     // so you'll need to resolve the borrow before running another query.
//!     let mut query: XPathModuleQuery<'_> = mtree.query();
//!
//!     // We can introspect on the current XPath expression being accumulated
//!     // by a query by calling `expr()`.
//!     assert_eq!(query.expr(), "/XmlModuleTree/Structure");
//!
//!     // [`XPathModuleQuery`] has a Debug impl:
//!     assert_eq!(
//!         format!("{:#?}", query),
//!         indoc::formatdoc! {r#"
//!                 XPathModuleQuery {{
//!                     tree: XmlModuleTree {{
//!                       <XmlModuleTree version="{version}">
//!                         <Structure>
//!                           <Linear id="n:1" class="struct">
//!                             <Param id="n:2" name="weight" param_id="{weight_id}" class="tensor" kind="Float" dtype="{weight_dtype}" shape="2 3" rank="2"/>
//!                             <Param id="n:3" name="bias" param_id="{bias_id}" class="tensor" kind="Float" dtype="{bias_dtype}" shape="3" rank="1"/>
//!                           </Linear>
//!                         </Structure>
//!                       </XmlModuleTree>
//!                     }},
//!                     expr: "/XmlModuleTree/Structure",
//!                 }}"#,
//!                 version=XML_MODULE_TREE_VERSION,
//!                 weight_id = weight_desc.param_id(),
//!                 weight_dtype = format!("{:?}", weight_desc.dtype()),
//!                 bias_id = bias_desc.param_id(),
//!                 bias_dtype = format!("{:?}", bias_desc.dtype()),
//!             }
//!     );
//!
//!     // We can collect the current query results as XML fragments.
//!     // This is primarily useful for debugging.
//!     //
//!     // Initially, this will be the root Module node.
//!     assert_eq!(
//!             &query.to_fragments(true)?,
//!             &[indoc::formatdoc! {r#"
//!                 <Structure>
//!                   <Linear id="n:1" class="struct">
//!                     <Param id="n:2" name="weight" param_id="{weight_id}" class="tensor" kind="Float" dtype="{weight_dtype}" shape="2 3" rank="2"/>
//!                     <Param id="n:3" name="bias" param_id="{bias_id}" class="tensor" kind="Float" dtype="{bias_dtype}" shape="3" rank="1"/>
//!                   </Linear>
//!                 </Structure>"#,
//!                 weight_id = weight_desc.param_id(),
//!                 weight_dtype = format!("{:?}", weight_desc.dtype()),
//!                 bias_id = bias_desc.param_id(),
//!                 bias_dtype = format!("{:?}", bias_desc.dtype()),
//!             },],
//!         );
//!
//!     // The [`XPathModuleQuery::params`] method selects all the `Param` elements
//!     // in the current subtree.
//!     let mut query = mtree.query().params();
//!     assert_eq!(
//!         query.expr(),
//!         "/XmlModuleTree/Structure/descendant-or-self::Param"
//!     );
//!     assert_eq!(
//!             &query.to_fragments(false)?,
//!             &[
//!                 format!(
//!                     r#"<Param id="n:2" name="weight" param_id="{weight_id}" class="tensor" kind="Float" dtype="{weight_dtype}" shape="2 3" rank="2"/>"#,
//!                     weight_id = weight_desc.param_id(),
//!                     weight_dtype = format!("{:?}", weight_desc.dtype()),
//!                 ),
//!                 format!(
//!                     r#"<Param id="n:3" name="bias" param_id="{bias_id}" class="tensor" kind="Float" dtype="{bias_dtype}" shape="3" rank="1"/>"#,
//!                     bias_id = bias_desc.param_id(),
//!                     bias_dtype = format!("{:?}", bias_desc.dtype()),
//!                 )
//!             ],
//!         );
//!
//!     // A full coverage of the XPath language cannot be included here.
//!     // For more details, see: <https://en.wikipedia.org/wiki/XPath>
//!
//!     // The structural elements start at '/XmlModuleTree/Structure/$Elem'.
//!     // But there's only every exactly one root node (currently).
//!     //
//!     // We can select this using either:
//!     // - the element selector (here, "Linear").
//!     // - the wildcard selector ('*').
//!     // - (a bunch of other, longer XPath operators).
//!     //
//!     // "Linear":
//!     // - Select the root 'Linear' node,
//!     let mut query = mtree.query().select("Linear");
//!     assert_eq!(query.expr(), "/XmlModuleTree/Structure/Linear");
//!     assert_eq!(
//!             &query.to_fragments(true)?,
//!             &[indoc::formatdoc! {r#"
//!                 <Linear id="n:1" class="struct">
//!                   <Param id="n:2" name="weight" param_id="{weight_id}" class="tensor" kind="Float" dtype="{weight_dtype}" shape="2 3" rank="2"/>
//!                   <Param id="n:3" name="bias" param_id="{bias_id}" class="tensor" kind="Float" dtype="{bias_dtype}" shape="3" rank="1"/>
//!                 </Linear>"#,
//!                 weight_id = weight_desc.param_id(),
//!                 weight_dtype = format!("{:?}", weight_desc.dtype()),
//!                 bias_id = bias_desc.param_id(),
//!                 bias_dtype = format!("{:?}", bias_desc.dtype()),
//!             },],
//!         );
//!
//!     // Here's the same thing using the wildcard selector:
//!     //
//!     // "*":
//!     // - Select the root's children, which is only the 'Linear' node
//!     let mut query = mtree.query().select("*");
//!     assert_eq!(query.expr(), "/XmlModuleTree/Structure/*");
//!     assert_eq!(
//!             &query.to_fragments(true)?,
//!             &[indoc::formatdoc! {r#"
//!                 <Linear id="n:1" class="struct">
//!                   <Param id="n:2" name="weight" param_id="{weight_id}" class="tensor" kind="Float" dtype="{weight_dtype}" shape="2 3" rank="2"/>
//!                   <Param id="n:3" name="bias" param_id="{bias_id}" class="tensor" kind="Float" dtype="{bias_dtype}" shape="3" rank="1"/>
//!                 </Linear>"#,
//!                 weight_id = weight_desc.param_id(),
//!                 weight_dtype = format!("{:?}", weight_desc.dtype()),
//!                 bias_id = bias_desc.param_id(),
//!                 bias_dtype = format!("{:?}", bias_desc.dtype()),
//!             },],
//!         );
//!
//!     // We can select specific names of structural elements by name,
//!     // using an attribute predicated `[@name='name']`.
//!     //
//!     // "Linear/*[@name='weight']":
//!     // - Select the root 'Linear' node,
//!     // - Select all the children of 'Linear',
//!     // - Filter those to elements with the attribte 'name' set to 'weight'.
//!     //
//!     // The "expr[predicate,...]" syntax is used to write filters,
//!     // the selected values in `{expr}' are restricted to those where all
//!     // of the predicates are true.
//!     let mut query = mtree.query().select("Linear/*[@name='weight']");
//!     assert_eq!(
//!         query.expr(),
//!         "/XmlModuleTree/Structure/Linear/*[@name='weight']"
//!     );
//!     assert_eq!(
//!             &query.to_fragments(false)?,
//!             &[format!(
//!                 r#"<Param id="n:2" name="weight" param_id="{weight_id}" class="tensor" kind="Float" dtype="{weight_dtype}" shape="2 3" rank="2"/>"#,
//!                 weight_id = weight_desc.param_id(),
//!                 weight_dtype = format!("{:?}", weight_desc.dtype()),
//!             ),],
//!         );
//!
//!     // We can also select the children of elements by their index.
//!     // Note: XPath indexes from 1, not 0.
//!     //
//!     // The children of sequences in the "bulitins" class ('Tuple', 'Vec', 'Array')
//!     // don't have names, but do have positional indices in XPath.
//!     //
//!     // "Linear/*[2]":
//!     // - Select the root 'Linear' node,
//!     // - Select all the children of 'Linear',
//!     // - Select the 2nd (indexing from 1) child.
//!     let mut query = mtree.query().select("Linear/*[2]");
//!     assert_eq!(query.expr(), "/XmlModuleTree/Structure/Linear/*[2]");
//!     assert_eq!(
//!             &query.to_fragments(false)?,
//!             &[format!(
//!                 r#"<Param id="n:3" name="bias" param_id="{bias_id}" class="tensor" kind="Float" dtype="{bias_dtype}" shape="3" rank="1"/>"#,
//!                 bias_id = bias_desc.param_id(),
//!                 bias_dtype = format!("{:?}", bias_desc.dtype()),
//!             )],
//!         );
//!
//!     // In general:
//!     // - `query.select(expr)` appends "/expr" to the current query expression.
//!     // - `query.filter(expr)` appends "[expr]" to the current query expression.
//!     //
//!     // `query.params()` may seem superflous, as both `.to_param_ids()` and
//!     // `.to_param_descs()` Implictly call `.params()`.
//!     //
//!     // However, when used in conjunction with `.filter()`, we can write powerful
//!     // selection expressions.
//!     let mut query = mtree.query().params().filter("@rank=2");
//!     assert_eq!(
//!         query.expr(),
//!         "/XmlModuleTree/Structure/descendant-or-self::Param[@rank=2]"
//!     );
//!     assert_eq!(
//!             &query.to_fragments(false)?,
//!             &[format!(
//!                 r#"<Param id="n:2" name="weight" param_id="{weight_id}" class="tensor" kind="Float" dtype="{weight_dtype}" shape="2 3" rank="2"/>"#,
//!                 weight_id = weight_desc.param_id(),
//!                 weight_dtype = format!("{:?}", weight_desc.dtype()),
//!             ),],
//!         );
//!
//!     // Many structural builtin components are also Modules.
//!     let module = (
//!         LinearConfig::new(2, 3).init::<B>(&device),
//!         [LinearConfig::new(4, 5).init::<B>(&device)],
//!         vec![
//!             LinearConfig::new(6, 7).init::<B>(&device),
//!             LinearConfig::new(8, 9).init::<B>(&device),
//!         ],
//!     );
//!     let expected_dtype = module.0.weight.dtype();
//!     let dtype_str = format!("{:?}", expected_dtype);
//!     // So we can still walk these modules:
//!     let mut mtree = XmlModuleTree::build(&module);
//!     assert_eq!(
//!             &mtree.query().to_fragments(true)?,
//!             &[indoc::formatdoc! {r#"
//!                 <Structure>
//!                   <Tuple id="n:1" class="builtin">
//!                     <Linear id="n:2" class="struct">
//!                       <Param id="n:3" name="weight" param_id="{}" class="tensor" kind="Float" dtype="{dtype}" shape="2 3" rank="2"/>
//!                       <Param id="n:4" name="bias" param_id="{}" class="tensor" kind="Float" dtype="{dtype}" shape="3" rank="1"/>
//!                     </Linear>
//!                     <Array id="n:5" class="builtin">
//!                       <Linear id="n:6" class="struct">
//!                         <Param id="n:7" name="weight" param_id="{}" class="tensor" kind="Float" dtype="{dtype}" shape="4 5" rank="2"/>
//!                         <Param id="n:8" name="bias" param_id="{}" class="tensor" kind="Float" dtype="{dtype}" shape="5" rank="1"/>
//!                       </Linear>
//!                     </Array>
//!                     <Vec id="n:9" class="builtin">
//!                       <Linear id="n:A" class="struct">
//!                         <Param id="n:B" name="weight" param_id="{}" class="tensor" kind="Float" dtype="{dtype}" shape="6 7" rank="2"/>
//!                         <Param id="n:C" name="bias" param_id="{}" class="tensor" kind="Float" dtype="{dtype}" shape="7" rank="1"/>
//!                       </Linear>
//!                       <Linear id="n:D" class="struct">
//!                         <Param id="n:E" name="weight" param_id="{}" class="tensor" kind="Float" dtype="{dtype}" shape="8 9" rank="2"/>
//!                         <Param id="n:F" name="bias" param_id="{}" class="tensor" kind="Float" dtype="{dtype}" shape="9" rank="1"/>
//!                       </Linear>
//!                     </Vec>
//!                   </Tuple>
//!                 </Structure>"#,
//!                 module.0.weight.id,
//!                 module.0.bias.as_ref().unwrap().id,
//!                 module.1[0].weight.id,
//!                 module.1[0].bias.as_ref().unwrap().id,
//!                 module.2[0].weight.id,
//!                 module.2[0].bias.as_ref().unwrap().id,
//!                 module.2[1].weight.id,
//!                 module.2[1].bias.as_ref().unwrap().id,
//!                 dtype=dtype_str,
//!             }],
//!         );
//!
//!     // We could select all the `Linear` descendants:
//!     let mut query = mtree.select("*//Linear");
//!     assert_eq!(query.expr(), "/XmlModuleTree/Structure/*//Linear");
//!     assert_eq!(
//!             &query.to_fragments(true)?,
//!             &[
//!                 indoc::formatdoc! {r#"
//!                   <Linear id="n:2" class="struct">
//!                     <Param id="n:3" name="weight" param_id="{}" class="tensor" kind="Float" dtype="{dtype}" shape="2 3" rank="2"/>
//!                     <Param id="n:4" name="bias" param_id="{}" class="tensor" kind="Float" dtype="{dtype}" shape="3" rank="1"/>
//!                   </Linear>"#,
//!                     module.0.weight.id,
//!                     module.0.bias.as_ref().unwrap().id,
//!                     dtype=dtype_str,
//!                 },
//!                 indoc::formatdoc! {r#"
//!                   <Linear id="n:6" class="struct">
//!                     <Param id="n:7" name="weight" param_id="{}" class="tensor" kind="Float" dtype="{dtype}" shape="4 5" rank="2"/>
//!                     <Param id="n:8" name="bias" param_id="{}" class="tensor" kind="Float" dtype="{dtype}" shape="5" rank="1"/>
//!                   </Linear>"#,
//!                     module.1[0].weight.id,
//!                     module.1[0].bias.as_ref().unwrap().id,
//!                     dtype=dtype_str,
//!                 },
//!                 indoc::formatdoc! {r#"
//!                   <Linear id="n:A" class="struct">
//!                     <Param id="n:B" name="weight" param_id="{}" class="tensor" kind="Float" dtype="{dtype}" shape="6 7" rank="2"/>
//!                     <Param id="n:C" name="bias" param_id="{}" class="tensor" kind="Float" dtype="{dtype}" shape="7" rank="1"/>
//!                   </Linear>"#,
//!                     module.2[0].weight.id,
//!                     module.2[0].bias.as_ref().unwrap().id,
//!                     dtype=dtype_str,
//!                 },
//!                 indoc::formatdoc! {r#"
//!                   <Linear id="n:D" class="struct">
//!                     <Param id="n:E" name="weight" param_id="{}" class="tensor" kind="Float" dtype="{dtype}" shape="8 9" rank="2"/>
//!                     <Param id="n:F" name="bias" param_id="{}" class="tensor" kind="Float" dtype="{dtype}" shape="9" rank="1"/>
//!                   </Linear>"#,
//!                     module.2[1].weight.id,
//!                     module.2[1].bias.as_ref().unwrap().id,
//!                     dtype=dtype_str,
//!                 },
//!             ],
//!         );
//!
//!     // Returning to the power of .params().filter("@rank=2"), we can see the effect
//!     // on complex modules:
//!     let mut query = mtree.query().params().filter("@rank=2");
//!     assert_eq!(
//!         query.expr(),
//!         "/XmlModuleTree/Structure/descendant-or-self::Param[@rank=2]"
//!     );
//!     assert_eq!(
//!             &query.to_fragments(false)?,
//!             &[
//!                 format!(
//!                     r#"<Param id="n:3" name="weight" param_id="{}" class="tensor" kind="Float" dtype="{dtype}" shape="2 3" rank="2"/>"#,
//!                     module.0.weight.id,
//!                     dtype = dtype_str,
//!                 ),
//!                 format!(
//!                     r#"<Param id="n:7" name="weight" param_id="{}" class="tensor" kind="Float" dtype="{dtype}" shape="4 5" rank="2"/>"#,
//!                     module.1[0].weight.id,
//!                     dtype = dtype_str,
//!                 ),
//!                 format!(
//!                     r#"<Param id="n:B" name="weight" param_id="{}" class="tensor" kind="Float" dtype="{dtype}" shape="6 7" rank="2"/>"#,
//!                     module.2[0].weight.id,
//!                     dtype = dtype_str,
//!                 ),
//!                 format!(
//!                     r#"<Param id="n:E" name="weight" param_id="{}" class="tensor" kind="Float" dtype="{dtype}" shape="8 9" rank="2"/>"#,
//!                     module.2[1].weight.id,
//!                     dtype = dtype_str,
//!                 ),
//!             ],
//!         );
//!
//!     Ok(())
//! }
//! ```

pub mod module_visitors;
pub mod xml_support;

mod xml_module_tree;
#[doc(inline)]
pub use xml_module_tree::*;
