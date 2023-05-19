import {
  Comparator,
  Comparators,
  Comparison,
  Operation,
  Operator,
  Operators,
  StructuredQuery,
  Visitor,
  VisitorComparisonResult,
  VisitorOperationResult,
  VisitorResult,
  VisitorStructuredQueryResult,
} from "../../chains/query_constructor/ir.js";

export type TranslatorOpts = {
  allowedOperators?: Operator[];
  allowedComparators?: Comparator[];
};

export abstract class BaseTranslator extends Visitor {
  allowedOperators: Operator[];

  allowedComparators: Comparator[];

  constructor(opts?: TranslatorOpts) {
    super();
    this.allowedOperators = opts?.allowedOperators ?? [
      Operators.and,
      Operators.or,
    ];
    this.allowedComparators = opts?.allowedComparators ?? [
      Comparators.eq,
      Comparators.neq,
      Comparators.gt,
      Comparators.gte,
      Comparators.lt,
      Comparators.lte,
      Comparators.in,
      Comparators.nin,
    ];
  }

  abstract formatFunction(func: Operator | Comparator): string;
}

export class BasicTranslator extends BaseTranslator {
  formatFunction(func: Operator | Comparator): string {
    if (func in Comparators) {
      if (
        this.allowedComparators.length > 0 &&
        this.allowedComparators.indexOf(func as Comparator) === -1
      ) {
        throw new Error(
          `Comparator ${func} not allowed. Allowed operators: ${this.allowedComparators.join(
            ", "
          )}`
        );
      }
    } else if (func in Operators) {
      if (
        this.allowedOperators.length > 0 &&
        this.allowedOperators.indexOf(func as Operator) === -1
      ) {
        throw new Error(
          `Operator ${func} not allowed. Allowed operators: ${this.allowedOperators.join(
            ", "
          )}`
        );
      }
    } else {
      throw new Error("Unknown comparator or operator");
    }
    return `$${func}`;
  }

  visitOperation(operation: Operation): VisitorOperationResult {
    const args = operation.args?.map((arg) =>
      arg.accept(this)
    ) as VisitorResult[];
    return {
      [this.formatFunction(operation.operator)]: args,
    };
  }

  visitComparison(comparison: Comparison): VisitorComparisonResult {
    return {
      [comparison.attribute]: {
        [this.formatFunction(comparison.comparator)]: comparison.value,
      },
    };
  }

  visitStructuredQuery(query: StructuredQuery): VisitorStructuredQueryResult {
    let nextArg = {};
    if (query.filter) {
      nextArg = {
        filter: query.filter.accept(this),
      };
    }
    return nextArg;
  }
}
