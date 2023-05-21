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

const reservedWords = [
  "break",
  "case",
  "catch",
  "class",
  "const",
  "continue",
  "debugger",
  "default",
  "delete",
  "do",
  "else",
  "enum",
  "export",
  "extends",
  "finally",
  "for",
  "function",
  "if",
  "import",
  "in",
  "instanceof",
  "new",
  "return",
  "super",
  "switch",
  "throw",
  "try",
  "typeof",
  "var",
  "void",
  "while",
  "with",
];
const reservedRegexes = reservedWords.map((word) => (input: string) => {
  const pattern = new RegExp(
    `(?<!['"]|\\$\\$)\\b${word}\\b(?!['"]|\\$\\$)`,
    "i"
  );
  const match = pattern.exec(input);
  return match ? { index: match.index, word } : null;
});

const splitOn = (slicable: string, ...indices: number[]) =>
  [0, ...indices].map((n, i, m) => slicable.slice(n, m[i + 1]));

export class ExpressionParser extends BaseOutputParser<ParsedType> {
  constructor(private allowReservedWords: boolean = true) {
    super();
  }

  async parse(text: string): Promise<ParsedType> {
    const parse = await ASTParser.importASTParser();
    let program: ESTree.Program;
    try {
      program = parse(text);
    } catch (err) {
      if (this.allowReservedWords) {
        for (const regex of reservedRegexes) {
          const regexResult = regex(text);
          if (regexResult !== null) {
            const { index, word } = regexResult;
            const split = splitOn(text, index, index + word.length);
            split[1] = `$$${split[1]}$$`;
            return this.parse(split.join(""));
          }
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
