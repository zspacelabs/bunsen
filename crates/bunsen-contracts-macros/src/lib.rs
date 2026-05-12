//! `proc_macro` support for BIMM Contracts.
#![no_std]
#![warn(missing_docs)]

#[cfg(test)]
#[macro_use]
extern crate std;

extern crate alloc;

use alloc::{
    boxed::Box,
    collections::BTreeSet,
    string::String,
    vec,
    vec::Vec,
};

use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use syn::{
    LitStr,
    Result as SynResult,
    Token,
    parse::{
        Parse,
        ParseStream,
    },
    parse_macro_input,
};

fn label_to_slot(
    label: &str,
    index: &[&str],
) -> usize {
    index.iter().position(|k| *k == label).unwrap()
}

/// Parse shape contract from token stream.
fn parse_shape_contract_terms(input: ParseStream) -> SynResult<ShapeContractAST> {
    let mut terms = Vec::new();

    while !input.is_empty() {
        // Parse a dim term
        let term = parse_dim_matcher_tokens(input)?;
        terms.push(term);

        // Check for comma separator
        if input.peek(Token![,]) {
            input.parse::<Token![,]>()?;
        } else {
            break;
        }
    }

    Ok(ShapeContractAST { terms })
}

/// Parse a single contract dim term from tokens.
fn parse_dim_matcher_tokens(input: ParseStream) -> SynResult<DimMatcherAST> {
    let mut label = None;

    // peek 2: ["name" =]
    if input.peek(LitStr) && input.peek2(Token![=]) {
        let lit: LitStr = input.parse()?;
        label = Some(lit.value());
        input.parse::<Token![=]>()?;
    }

    // Check for "_" (underscore) for Any
    if input.peek(Token![_]) {
        input.parse::<Token![_]>()?;
        return Ok(DimMatcherAST::Any { label });
    }

    // Check for ellipsis "..."
    if input.peek(Token![...]) {
        input.parse::<Token![...]>()?;
        return Ok(DimMatcherAST::Ellipsis { label });
    }

    // Otherwise, parse as an expression.
    let expr = parse_expr_tokens(input)?;
    Ok(DimMatcherAST::Expr { label, expr })
}

/// Represents a node in a dimension expression.
#[derive(Debug, Clone, PartialEq)]
enum ExprAST {
    /// A parameter (string literal).
    Param(String),

    /// Negation of another expression.
    Negate(Box<ExprAST>),

    /// Power of an expression (base raised to an exponent).
    Pow(Box<ExprAST>, usize),

    /// Sum of multiple expressions (addition and subtraction).
    Sum(Vec<ExprAST>),

    /// Product of multiple expressions (multiplication).
    Prod(Vec<ExprAST>),
}

impl ExprAST {
    /// Add all parameter labels to the set.
    pub fn collect_labels(
        &self,
        labels: &mut BTreeSet<String>,
    ) {
        match self {
            ExprAST::Param(name) => {
                labels.insert(name.clone());
            }
            ExprAST::Negate(expr) => {
                expr.collect_labels(labels);
            }
            ExprAST::Pow(base, _) => {
                base.collect_labels(labels);
            }
            ExprAST::Sum(terms) => {
                for term in terms {
                    term.collect_labels(labels);
                }
            }
            ExprAST::Prod(factors) => {
                for factor in factors {
                    factor.collect_labels(labels);
                }
            }
        }
    }
}

/// Represents a matcher for a dimension in a shape contract.
#[derive(Debug, Clone, PartialEq)]
enum DimMatcherAST {
    /// Matches any dimension, ignoring size.
    Any { label: Option<String> },

    /// Matches an ellipsis, allowing any number of dimensions.
    ///
    /// There can only be one ellipsis in a shape contract.
    Ellipsis { label: Option<String> },

    /// Matches a dimension based on an expression.
    Expr {
        label: Option<String>,
        expr: ExprAST,
    },
}

impl DimMatcherAST {
    pub fn label(&self) -> Option<String> {
        match self {
            DimMatcherAST::Any { label } => label.clone(),
            DimMatcherAST::Ellipsis { label } => label.clone(),
            DimMatcherAST::Expr { label, .. } => label.clone(),
        }
    }

    /// Add all parameter labels to the set.
    pub fn collect_labels(
        &self,
        labels: &mut BTreeSet<String>,
    ) {
        if let Some(label) = self.label() {
            labels.insert(label);
        }
        if let DimMatcherAST::Expr { expr, .. } = self {
            expr.collect_labels(labels);
        }
    }
}

/// Represents a shape contract, which consists of multiple dimension matchers.
///
/// The `shape_contract!` macro allows you to define a shape contract
/// that can be used to match shapes in a type-safe manner.
#[derive(Debug, Clone, PartialEq)]
struct ShapeContractAST {
    /// The terms of the shape contract, each represented by a `DimMatcher`.
    pub terms: Vec<DimMatcherAST>,
}

impl ShapeContractAST {
    /// Add all parameter labels to the set.
    pub fn collect_labels(
        &self,
        labels: &mut BTreeSet<String>,
    ) {
        for term in &self.terms {
            term.collect_labels(labels);
        }
    }

    /// Get all parameter labels.
    pub fn labels(&self) -> Vec<String> {
        let mut labels = BTreeSet::new();
        self.collect_labels(&mut labels);
        labels.into_iter().collect()
    }
}

/// Custom parser for shape contract syntax.
struct ContractSyntax {
    /// The parsed shape contract.
    contract: ShapeContractAST,
}

impl Parse for ContractSyntax {
    fn parse(input: ParseStream) -> SynResult<Self> {
        let contract = parse_shape_contract_terms(input)?;
        Ok(ContractSyntax { contract })
    }
}

/// Parse expression from token stream.
fn parse_expr_tokens(input: ParseStream) -> SynResult<ExprAST> {
    parse_sum_expr(input)
}

/// Parse sum/difference (lowest precedence).
fn parse_sum_expr(input: ParseStream) -> SynResult<ExprAST> {
    let left = parse_prod_expr(input)?;
    let mut terms = vec![left.clone()];

    while input.peek(Token![+]) || input.peek(Token![-]) {
        if input.parse::<Token![+]>().is_ok() {
            terms.push(parse_prod_expr(input)?);
        } else if input.parse::<Token![-]>().is_ok() {
            terms.push(ExprAST::Negate(Box::new(parse_prod_expr(input)?)));
        }
    }

    Ok(if terms.len() == 1 {
        terms.into_iter().next().unwrap()
    } else {
        ExprAST::Sum(terms)
    })
}

/// Parse multiplication (higher precedence).
fn parse_prod_expr(input: ParseStream) -> SynResult<ExprAST> {
    let mut factors = vec![parse_power_expr(input)?];

    while input.peek(Token![*]) {
        input.parse::<Token![*]>()?;
        factors.push(parse_power_expr(input)?);
    }

    Ok(if factors.len() == 1 {
        factors.into_iter().next().unwrap()
    } else {
        ExprAST::Prod(factors)
    })
}

/// Parse power (highest precedence).
fn parse_power_expr(input: ParseStream) -> SynResult<ExprAST> {
    let base = parse_factor_expr(input)?;

    if input.peek(Token![^]) {
        input.parse::<Token![^]>()?;
        let exp: syn::LitInt = input.parse()?;
        let exp_value: usize = exp.base10_parse()?;
        Ok(ExprAST::Pow(Box::new(base), exp_value))
    } else {
        Ok(base)
    }
}

/// Parse factors (parameters, parentheses, negation).
fn parse_factor_expr(input: ParseStream) -> SynResult<ExprAST> {
    // Handle unary operators
    if input.peek(Token![-]) {
        input.parse::<Token![-]>()?;
        let expr = parse_factor_expr(input)?;
        return Ok(ExprAST::Negate(Box::new(expr)));
    }

    if input.peek(Token![+]) {
        input.parse::<Token![+]>()?;
        // Unary plus is no-op
        return parse_factor_expr(input);
    }

    // Handle parentheses
    if input.peek(syn::token::Paren) {
        let content;
        syn::parenthesized!(content in input);
        return parse_expr_tokens(&content);
    }

    // Handle string literals (parameters)
    if input.peek(LitStr) {
        let lit: LitStr = input.parse()?;
        return Ok(ExprAST::Param(lit.value()));
    }

    Err(syn::Error::new(
        input.span(),
        "Expected parameter, parentheses, or unary operator",
    ))
}

impl ExprAST {
    fn to_tokens(
        &self,
        index: &[&str],
    ) -> TokenStream2 {
        match self {
            ExprAST::Param(name) => {
                let param_id = label_to_slot(name, index);
                quote! {
                    DimExpr::Param{id: #param_id}
                }
            }
            ExprAST::Negate(expr) => {
                let inner = expr.to_tokens(index);
                quote! {
                    DimExpr::Negate{child: &#inner}
                }
            }
            ExprAST::Pow(base, exp) => {
                let base_tokens = base.to_tokens(index);
                quote! {
                    DimExpr::Pow{base: &#base_tokens, exp : #exp}
                }
            }
            ExprAST::Sum(terms) => {
                let term_tokens: Vec<_> = terms.iter().map(|t| t.to_tokens(index)).collect();
                quote! {
                    DimExpr::Sum{children: &[#(#term_tokens),*]}
                }
            }
            ExprAST::Prod(factors) => {
                let factor_tokens: Vec<_> = factors.iter().map(|f| f.to_tokens(index)).collect();
                quote! {
                    DimExpr::Prod{children: &[#(#factor_tokens),*]}
                }
            }
        }
    }
}

impl DimMatcherAST {
    fn to_tokens(
        &self,
        index: &[&str],
    ) -> TokenStream2 {
        match self {
            DimMatcherAST::Any { label } => {
                let base = quote! { DimMatcher::any() };
                if let Some(label) = label {
                    let label_id = label_to_slot(label, index);
                    quote! { #base.with_label_id(Some(#label_id)) }
                } else {
                    base
                }
            }
            DimMatcherAST::Ellipsis { label } => {
                let base = quote! { DimMatcher::ellipsis() };
                if let Some(label) = label {
                    let label_id = label_to_slot(label, index);
                    quote! { #base.with_label_id(Some(#label_id)) }
                } else {
                    base
                }
            }
            DimMatcherAST::Expr { label, expr } => {
                let expr_tokens = expr.to_tokens(index);
                let base = quote! { DimMatcher::expr(#expr_tokens) };
                if let Some(label) = label {
                    let label_id = label_to_slot(label, index);
                    quote! { #base.with_label_id(Some(#label_id)) }
                } else {
                    base
                }
            }
        }
    }
}

impl ShapeContractAST {
    fn to_tokens(&self) -> TokenStream2 {
        let labels = self.labels();
        let index: Vec<&str> = labels.iter().map(|label| label.as_str()).collect();
        let index_tokens = index
            .iter()
            .map(|label| quote! { #label })
            .collect::<Vec<_>>();
        let term_tokens: Vec<_> = self.terms.iter().map(|t| t.to_tokens(&index)).collect();
        quote! {
            ShapeContract::new(
                &[#(#index_tokens),*],
                &[#(#term_tokens),*],
            )
        }
    }
}

/// Parse a shape contract at compile time and return the `ShapePattern` struct.
///
/// This macro generates `no_std` compatible code.
///
/// A shape pattern is made of one or more dimension matcher terms:
/// - `_`: for any shape; ignores the size, but requires the dimension to
///   exist.,
/// - `...`: for ellipsis; matches any number of dimensions, only one ellipsis
///   is allowed,
/// - a dim expression.
///
/// ```bnf
/// ShapeContract => <LabeledExpr> { ',' <LabeledExpr> }* ','?
/// LabeledExpr => {Param "="}? <Expr>
/// Expr => <Term> { <AddOp> <Term> }
/// Term => <Power> { <MulOp> <Power> }
/// Power => <Factor> [ ^ <usize> ]
/// Factor => <Param> | ( '(' <Expression> ')' ) | NegOp <Factor>
/// Param => '"' <identifier> '"'
/// identifier => { <alpha> | "_" } { <alphanumeric> | "_" }*
/// NegOp =>      '+' | '-'
/// AddOp =>      '+' | '-'
/// MulOp =>      '*'
/// ```
///
/// # Example
/// ```rust.norun
/// use bunsen_contracts::{ShapeContract, shape_contract};
/// static CONTRACT: ShapeContract = shape_contract![_, "x" + "y", ..., "z" ^ 2];
/// ```
#[proc_macro]
pub fn shape_contract(input: TokenStream) -> TokenStream {
    let parsed = parse_macro_input!(input as ContractSyntax);
    let tokens = parsed.contract.to_tokens();
    quote! {
        {
            extern crate alloc;
            #[allow(unused_imports)]
            use bunsen_contracts::{ShapeContract, DimMatcher, DimExpr};
            use alloc::boxed::Box;
            #tokens
        }
    }
    .into()
}

#[cfg(test)]
mod tests {
    use alloc::{
        string::ToString,
        vec,
    };

    use super::*;

    struct ExprSyntax {
        expr: ExprAST,
    }

    impl Parse for ExprSyntax {
        fn parse(input: ParseStream) -> SynResult<Self> {
            let expr = parse_expr_tokens(input)?;
            Ok(ExprSyntax { expr })
        }
    }

    fn assert_token_stream_eq(
        actual: &TokenStream2,
        expected: &TokenStream2,
    ) {
        // It would be nice to pretty print on error.
        assert_eq!(actual.to_string(), expected.to_string())
    }

    #[test]
    fn test_unary_add_op() {
        let tokens: proc_macro2::TokenStream = r#"+ "x""#.parse().unwrap();
        let input = syn::parse2::<ExprSyntax>(tokens).unwrap();
        assert_eq!(input.expr, ExprAST::Param("x".to_string()));

        let index = ["x"];
        assert_token_stream_eq(
            &input.expr.to_tokens(&index),
            &quote! {
            DimExpr::Param { id: 0usize }
            },
        );
    }

    #[test]
    fn test_parse_simple_expression() {
        let tokens: proc_macro2::TokenStream = r#""x""#.parse().unwrap();
        let input = syn::parse2::<ExprSyntax>(tokens).unwrap();
        assert_eq!(input.expr, ExprAST::Param("x".to_string()));

        let index = ["x"];
        assert_eq!(
            input.expr.to_tokens(&index).to_string(),
            "DimExpr :: Param { id : 0usize }"
        );
    }

    #[test]
    fn test_parse_unary_negation() {
        let tokens: proc_macro2::TokenStream = r#"-"x""#.parse().unwrap();
        let input = syn::parse2::<ExprSyntax>(tokens).unwrap();
        assert_eq!(
            input.expr,
            ExprAST::Negate(Box::new(ExprAST::Param("x".to_string())))
        );

        let index = ["x"];
        assert_token_stream_eq(
            &input.expr.to_tokens(&index),
            &quote! {
            DimExpr::Negate{ child:
                &DimExpr::Param { id: 0usize } }
            },
        );
    }

    #[test]
    fn test_mixed_addition() {
        let tokens: proc_macro2::TokenStream = r#""a" + "b" - "x""#.parse().unwrap();
        let input = syn::parse2::<ExprSyntax>(tokens).unwrap();
        assert_eq!(
            input.expr,
            ExprAST::Sum(vec![
                ExprAST::Param("a".to_string()),
                ExprAST::Param("b".to_string()),
                ExprAST::Negate(Box::new(ExprAST::Param("x".to_string()))),
            ])
        );
    }

    #[test]
    fn test_mangled_addition() {
        let tokens: proc_macro2::TokenStream = r#""a" + - - "x""#.parse().unwrap();
        let input = syn::parse2::<ExprSyntax>(tokens).unwrap();
        assert_eq!(
            input.expr,
            ExprAST::Sum(vec![
                ExprAST::Param("a".to_string()),
                ExprAST::Negate(Box::new(ExprAST::Negate(Box::new(ExprAST::Param(
                    "x".to_string()
                ))))),
            ])
        );

        let index = ["x", "a"];
        assert_eq!(
            input.expr.to_tokens(&index).to_string(),
            "DimExpr :: Sum { children : & [DimExpr :: Param { id : 1usize } , DimExpr :: Negate { child : & DimExpr :: Negate { child : & DimExpr :: Param { id : 0usize } } }] }"
        );
    }

    #[test]
    fn test_parse_power_precedence() {
        let tokens: proc_macro2::TokenStream = r#"-"x" ^ 3"#.parse().unwrap();
        let input = syn::parse2::<ExprSyntax>(tokens).unwrap();
        assert_eq!(
            input.expr,
            ExprAST::Pow(
                Box::new(ExprAST::Negate(Box::new(ExprAST::Param("x".to_string())))),
                3
            )
        );

        let index = ["x", "a"];
        assert_eq!(
            input.expr.to_tokens(&index).to_string(),
            "DimExpr :: Pow { base : & DimExpr :: Negate { child : & DimExpr :: Param { id : 0usize } } , exp : 3usize }"
        );
    }

    #[test]
    fn test_parse_shape_contract_terms() {
        let tokens: proc_macro2::TokenStream =
            r#""any" = _, "x", ..., "y" + ("z" * "w") ^ 2"#.parse().unwrap();
        let input = syn::parse2::<ContractSyntax>(tokens).unwrap();
        let contract = input.contract;

        assert_eq!(contract.terms.len(), 4);
        assert_eq!(
            contract.terms[0],
            DimMatcherAST::Any {
                label: Some("any".to_string())
            }
        );
        assert_eq!(
            contract.terms[1],
            DimMatcherAST::Expr {
                label: None,
                expr: ExprAST::Param("x".to_string())
            }
        );
        assert_eq!(contract.terms[2], DimMatcherAST::Ellipsis { label: None });
        assert_eq!(
            contract.terms[3],
            DimMatcherAST::Expr {
                label: None,
                expr: ExprAST::Sum(vec![
                    ExprAST::Param("y".to_string()),
                    ExprAST::Pow(
                        Box::new(ExprAST::Prod(vec![
                            ExprAST::Param("z".to_string()),
                            ExprAST::Param("w".to_string())
                        ])),
                        2
                    ),
                ])
            }
        );

        // any, w, x, y, z
        assert_token_stream_eq(
            &contract.to_tokens(),
            &quote! {
            ShapeContract::new(
                &["any", "w", "x", "y", "z"],
                &[
                    DimMatcher::any().with_label_id(Some(0usize)),
                    DimMatcher::expr(
                        DimExpr::Param { id: 2usize }),
                    DimMatcher::ellipsis(),
                    DimMatcher::expr(
                        DimExpr::Sum{
                            children: &[
                                DimExpr::Param { id: 3usize },
                                DimExpr::Pow {
                                    base: &DimExpr::Prod {
                                        children: &[
                                            DimExpr::Param { id: 4usize },
                                            DimExpr::Param { id: 1usize }
                                        ]
                                    },
                                    exp: 2usize
                                }]
                        })
                ],
            )},
        );
    }
}
