const SEQUENCE_TYPES: &[&str] = &["Vec", "Array", "Tuple"];

/// Is this `container_type` a sequence type?
///
/// See: [`burn::module::ModuleVisitor::enter_module`].
pub fn container_type_is_sequence(container_type: &str) -> bool {
    SEQUENCE_TYPES.contains(&container_type)
}

/// Parse the container type into (class, name).
///
/// See: [`burn::module::ModuleVisitor::enter_module`].
///
/// "Vec" => ("builtin", "Vec")
/// "Struct:Foo" => ("struct", "Foo")
/// "Enum:Foo" => ("enum", "Foo")
pub fn parse_container_type(container_type: &str) -> (String, String) {
    if let Some((cls, name)) = container_type.split_once(':') {
        let cls = cls.to_lowercase();
        (cls, name.to_string())
    } else {
        ("builtin".to_string(), container_type.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_type_is_sequence() {
        assert!(container_type_is_sequence("Vec"));
        assert!(container_type_is_sequence("Array"));
        assert!(container_type_is_sequence("Tuple"));

        assert!(!container_type_is_sequence("Thingy"));

        assert!(!container_type_is_sequence("Struct:Foo"));
        assert!(!container_type_is_sequence("Enum:Foo"));
    }

    #[test]
    fn test_parse_container_type() {
        assert_eq!(
            parse_container_type("Vec"),
            ("builtin".to_string(), "Vec".to_string())
        );
        assert_eq!(
            parse_container_type("Struct:Foo"),
            ("struct".to_string(), "Foo".to_string())
        );
        assert_eq!(
            parse_container_type("Enum:Foo"),
            ("enum".to_string(), "Foo".to_string())
        );
    }
}
