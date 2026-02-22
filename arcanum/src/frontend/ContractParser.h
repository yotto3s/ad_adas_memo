#pragma once

#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/RawCommentList.h"

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace arcanum {

// ============================================================
// Contract Expression AST
// ============================================================

/// Forward declaration for recursive types.
struct ContractExpr;
using ContractExprPtr = std::unique_ptr<ContractExpr>;

/// Binary operators supported in contract expressions.
enum class BinOp {
    Add,      // +
    Sub,      // -
    Mul,      // *
    Div,      // /
    Mod,      // %
    Lt,       // <
    Le,       // <=
    Gt,       // >
    Ge,       // >=
    Eq,       // ==
    Ne,       // !=
    And,      // &&
    Or,       // ||
};

/// Unary operators supported in contract expressions.
enum class UnaryOp {
    Not,   // !
    Neg,   // - (unary minus)
};

/// An integer literal in a contract expression.
struct IntLiteral {
    int64_t value;
};

/// A boolean literal in a contract expression.
struct BoolLiteral {
    bool value;
};

/// A reference to a function parameter or variable by name.
struct Identifier {
    std::string name;
};

/// A reference to the return value (\result).
struct ResultRef {};

/// A binary operation in a contract expression.
struct BinaryExpr {
    BinOp op;
    ContractExprPtr lhs;
    ContractExprPtr rhs;
};

/// A unary operation in a contract expression.
struct UnaryExpr {
    UnaryOp op;
    ContractExprPtr operand;
};

/// A contract expression node (variant-based AST).
struct ContractExpr {
    std::variant<IntLiteral, BoolLiteral, Identifier, ResultRef, BinaryExpr,
                 UnaryExpr>
        node;
};

/// Helper constructors for building contract expression AST nodes.
ContractExprPtr makeIntLiteral(int64_t value);
ContractExprPtr makeBoolLiteral(bool value);
ContractExprPtr makeIdentifier(std::string name);
ContractExprPtr makeResultRef();
ContractExprPtr makeBinaryExpr(BinOp op, ContractExprPtr lhs,
                               ContractExprPtr rhs);
ContractExprPtr makeUnaryExpr(UnaryOp op, ContractExprPtr operand);

/// Render a contract expression to a human-readable string (for debugging).
std::string contractExprToString(const ContractExpr &expr);

// ============================================================
// Contract Annotation Types
// ============================================================

/// The type of a contract annotation.
enum class ContractKind {
    Requires, // //@ requires: <expr>
    Ensures,  // //@ ensures: <expr>
};

/// A single parsed contract annotation.
struct ContractAnnotation {
    ContractKind kind;
    ContractExprPtr expr;
    unsigned sourceLine; // Source line where the annotation was found
};

/// All contract annotations attached to a single function.
struct ContractInfo {
    std::vector<ContractAnnotation> preconditions;  // requires clauses
    std::vector<ContractAnnotation> postconditions; // ensures clauses

    bool empty() const {
        return preconditions.empty() && postconditions.empty();
    }
};

// ============================================================
// Parse errors
// ============================================================

/// An error encountered during contract parsing.
struct ContractParseError {
    unsigned sourceLine;
    unsigned column;
    std::string message;
};

// ============================================================
// Contract Expression Parser (standalone, for testing)
// ============================================================

/// Result of parsing a contract expression string.
struct ExprParseResult {
    ContractExprPtr expr;
    std::vector<ContractParseError> errors;

    bool success() const { return expr != nullptr && errors.empty(); }
};

/// Parse a contract expression from a string.
/// This is the core expression parser used internally by ContractParser,
/// exposed here for direct unit testing.
ExprParseResult parseContractExpr(const std::string &source,
                                  unsigned sourceLine = 0);

// ============================================================
// ContractParser
// ============================================================

/// Result of running the contract parser on a translation unit.
struct ContractParserResult {
    /// Map from function declaration to its parsed contracts.
    /// Uses a vector of pairs since we cannot hash FunctionDecl*.
    std::vector<std::pair<const clang::FunctionDecl *, ContractInfo>> contracts;

    /// All parse errors encountered.
    std::vector<ContractParseError> errors;

    bool hasErrors() const { return !errors.empty(); }

    /// Find the contract info for a given function, or nullptr if none.
    const ContractInfo *findContract(const clang::FunctionDecl *fn) const;
};

/// Parses //@ contract annotations from Clang comments and associates
/// them with the corresponding function declarations.
///
/// For Slice 1, supports:
///   - //@ requires: <expr>   -- precondition
///   - //@ ensures: <expr>    -- postcondition
///   - \result                -- return value reference in ensures
///
/// Expression language supports:
///   - Arithmetic: +, -, *, /, %
///   - Comparison: <, <=, >, >=, ==, !=
///   - Logical: &&, ||, !
///   - Identifiers (function parameter names)
///   - Integer literals
///   - Boolean literals (true, false)
///   - Parenthesized sub-expressions
class ContractParser {
public:
    explicit ContractParser(clang::ASTContext &context);

    /// Parse all //@ annotations in the translation unit and associate
    /// them with their corresponding function declarations.
    ContractParserResult parse();

private:
    clang::ASTContext &context_;

    /// Extract the raw text from a comment, stripping the // prefix.
    std::string extractCommentText(const clang::RawComment *comment) const;

    /// Check if a comment line is a contract annotation (starts with //@).
    bool isContractComment(llvm::StringRef text) const;

    /// Parse a single annotation line (after stripping the //@ prefix).
    /// Returns a ContractAnnotation if valid, or adds errors.
    std::optional<ContractAnnotation>
    parseAnnotationLine(llvm::StringRef line, unsigned sourceLine,
                        std::vector<ContractParseError> &errors);

    /// Find the FunctionDecl that a comment is associated with.
    const clang::FunctionDecl *
    findAssociatedFunction(const clang::RawComment *comment) const;
};

} // namespace arcanum
