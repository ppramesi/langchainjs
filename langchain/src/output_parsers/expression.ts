import type { ESTree } from "meriyah";
import { MasterHandler } from "./expression_type_handlers/factory.js";
import { ParsedType } from "./expression_type_handlers/types.js";
import { BaseOutputParser } from "../schema/output_parser.js";
import { ASTParser } from "./expression_type_handlers/base.js";
/**
 * okay so we need to be able to handle the following cases:
 * ExpressionStatement
 *  CallExpression
 *      Identifier | MemberExpression
 *      ExpressionLiterals: [
 *          CallExpression
 *          StringLiteral
 *          NumericLiteral
 *          ArrayLiteralExpression
 *              ExpressionLiterals
 *          ObjectLiteralExpression
 *              PropertyAssignment
 *                  Identifier
 *                  ExpressionLiterals
 *      ]
 */

type ParserError = {
  index: number;
  line: number;
  column: number;
  description: string;
  loc: {
    line: number;
    column: number;
  };
};

const reservedWords = [
  "abstract",
  "arguments",
  "await",
  "boolean",
  "break",
  "byte",
  "case",
  "catch",
  "char",
  "class",
  "const",
  "continue",
  "debugger",
  "default",
  "delete",
  "do",
  "double",
  "else",
  "enum",
  "eval",
  "export",
  "extends",
  "false",
  "final",
  "finally",
  "float",
  "for",
  "function",
  "goto",
  "if",
  "implements",
  "import",
  "in",
  "instanceof",
  "int",
  "interface",
  "let",
  "long",
  "native",
  "new",
  "null",
  "package",
  "private",
  "protected",
  "public",
  "return",
  "short",
  "static",
  "super",
  "switch",
  "synchronized",
  "this",
  "throw",
  "throws",
  "transient",
  "true",
  "try",
  "typeof",
  "var",
  "void",
  "volatile",
  "while",
  "with",
  "yield",
];
const splitOn = (slicable: string, ...indices: number[]) =>
  [0, ...indices].map((n, i, m) => slicable.slice(n, m[i + 1]));

export class ExpressionParser extends BaseOutputParser<ParsedType> {
  async parse(text: string): Promise<ParsedType> {
    const parse = await ASTParser.importASTParser();
    let program: ESTree.Program;
    try {
      program = parse(text);
    } catch (err) {
      const error = err as ParserError;
      if (error.description.includes("Unexpected token: ")) {
        const pattern = /\[(\d+):(\d+)\]: Unexpected token: '(.*)'/;
        const match = error.description.match(pattern);
        if (match && match[3] && reservedWords.includes(match[3])) {
          const split = splitOn(
            text,
            error.column - match[3].length,
            error.column
          );
          split[1] = `$$${split[1]}$$`;
          return this.parse(split.join(""));
        }
      }
      throw new Error(`Error parsing ${err}: ${text}`);
    }

    if (program.body.length > 1) {
      throw new Error(`Expected 1 statement, got ${program.body.length}`);
    }

    const [node] = program.body;
    if (!ASTParser.isExpressionStatement(node)) {
      throw new Error(
        `Expected ExpressionStatement, got ${(node as ESTree.Node).type}`
      );
    }

    const { expression: expressionStatement } = node;
    if (!ASTParser.isCallExpression(expressionStatement)) {
      throw new Error("Expected CallExpression");
    }
    const masterHandler = MasterHandler.createMasterHandler();
    return await masterHandler.handle(expressionStatement);
  }

  getFormatInstructions(): string {
    return "";
  }
}

export * from "./expression_type_handlers/types.js";

export { MasterHandler } from "./expression_type_handlers/factory.js";
