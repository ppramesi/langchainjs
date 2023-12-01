/* eslint-disable no-loop-func */
import pgPromise, { IBaseProtocol, IDatabase } from "pg-promise";
import { MaxMarginalRelevanceSearchOptions, VectorStore } from "./base.js";
import { Embeddings } from "../embeddings/base.js";
import { Document } from "../document.js";
import { maximalMarginalRelevance } from "../util/math.js";
import { isFloat, isInt, isString } from "../util/type_utils.js";

class Fragment {
  constructor(
    public readonly query: string,
    public readonly values?: unknown[]
  ) {}

  static combineFragments(
    fragments: (Fragment | string | null)[],
    joiner = " ",
    enclosed = false
  ): Fragment {
    const queryArray = [];
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    let combinedValues: any[] = [];
    let placeholderOffset = 0;

    for (const fragment of fragments) {
      if (fragment == null) continue;
      // eslint-disable-next-line no-instanceof/no-instanceof
      if (fragment instanceof Fragment) {
        if (!fragment.query || fragment.query.length === 0) continue;
        let reparametrizedQuery = fragment.query;

        if (fragment.values && fragment.values.length > 0) {
          reparametrizedQuery = fragment.query.replace(
            /\$(\d+)/g,
            (_, n) => `$${parseInt(n, 10) + placeholderOffset}`
          );

          combinedValues = combinedValues.concat(fragment.values);
          placeholderOffset += fragment.values.length;
        }

        queryArray.push(reparametrizedQuery);
      } else {
        if (!fragment || fragment.length === 0) continue;
        queryArray.push(fragment);
      }
    }

    const newQuery = `${enclosed ? "(" : ""}${queryArray.join(joiner).trim()}${
      enclosed ? ")" : ""
    }`;

    return new Fragment(
      newQuery,
      combinedValues.length > 0 ? combinedValues : undefined
    );
  }
}

export type FilterValue = string | number;

/**
 * pgp is a factory. It's pure (I think).
 */
const pgp = /* #__PURE__ */ pgPromise();

export type TextSearchValue = {
  query: string;
  type?: "plain" | "phrase" | "websearch";
  config?: string;
};

const allowedOperators = ["=", "<>", ">", ">=", "<", "<="] as const;

export type OnCondition = {
  left: string;
  right: string;
  operator?: "=" | "<>" | ">" | ">=" | "<" | "<=";
};

export type JoinStatement = {
  op:
    | "JOIN"
    | "LEFT JOIN"
    | "RIGHT JOIN"
    | "FULL JOIN"
    | "CROSS JOIN"
    | "INNER JOIN";
  table: string;
  on: OnCondition[]; // array to support multiple conditions
};

export type ComparisonOperator =
  | { $eq: FilterValue }
  | { $gt: FilterValue }
  | { $gte: FilterValue }
  | { $lt: FilterValue }
  | { $lte: FilterValue }
  | { $not: FilterValue }
  | { $textSearch: TextSearchValue };

export type LogicalOperator = { $and: PGFilter[] } | { $or: PGFilter[] };

export type KeyValueFilter = {
  [key: string]: FilterValue | ComparisonOperator;
};

export type PGFilter = KeyValueFilter | LogicalOperator;

export type PGFilterWithJoin =
  | {
      metadataFilter?: never;
      columnFilter: PGFilter;
      join?: JoinStatement | JoinStatement[];
    }
  | {
      metadataFilter: PGFilter;
      columnFilter?: never;
      join?: JoinStatement | JoinStatement[];
    };

export type Metric = "cosine" | "l2" | "manhattan" | "inner_product";

interface CoreColumns {
  id: string;
  pageContentColumn: string;
  metadata: Record<string, unknown>;
  embedding: number[]; // Replace with the actual data type you use
}

// Use mapped types for extra columns
type ExtraColumns<T extends Record<string, unknown>> = {
  [K in keyof T]: T[K];
};

// Combine core and extra columns
type TableShape<T extends Record<string, unknown> = Record<string, unknown>> =
  CoreColumns & ExtraColumns<T>;

export type ExtensionOpts<T extends Record<string, unknown>> = {
  /**
   * The metric to use for similarity search.
   */
  metric?: Metric;

  /**
   * The number of dimensions of the embeddings.
   */
  dims?: number;

  /**
   * The pg-promise client to use.
   */
  pgDb: IBaseProtocol<TableShape<T>>;
};

const ComparisonMap = {
  $eq: "=",
  $lt: "<",
  $lte: "<=",
  $gt: ">",
  $gte: ">=",
  $not: "<>",
  $textSearch: "@@",
} as const;
type ComparisonMapKey = keyof typeof ComparisonMap;

const LogicalMap = {
  $and: "AND",
  $or: "OR",
} as const;
type LogicalMapKey = keyof typeof LogicalMap;

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type ColumnValue = { [K: string]: any };

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type PGQuery<R> = (t: IBaseProtocol<any>) => Promise<R>;

export type HNSWIndexStatementOpts = {
  /**
   * the max number of connections per layer (16 by default)
   */
  m?: number;

  /**
   * the size of the dynamic candidate list for constructing the graph (64 by default)
   */
  efConstruction?: number;

  /**
   * Influences the trade-off between query accuracy (recall) and speed. Only pgembedding supports this option.
   */
  efSearch?: number;
};

/**
 * Abstract class for Postgres embedding extensions. This class is used to
 * specify the Postgres extension to use for similarity search. The default
 * extension is pgvector, but pgembedding can also be used.
 */
export abstract class PostgresEmbeddingExtension<
  Columns extends Record<string, unknown> = Record<string, unknown>
> {
  selectedMetric: Metric;

  /**
   * The number of dimensions of the embeddings.
   */
  dims: number;

  pgInstance: IBaseProtocol<TableShape<Columns>>;

  constructor(extensionOpts: ExtensionOpts<Columns>) {
    const metric = extensionOpts.metric ?? "cosine";
    this.validateSelectedMetric(metric);
    this.selectedMetric = metric;
    this.dims = extensionOpts.dims ?? 1536; // defaults to OpenAI 1536 embedding dims
    this.pgInstance = extensionOpts.pgDb;
  }

  abstract allowedMetrics(): Metric[];

  /**
   * Validate the selected metric. Check if it is one of the allowed metrics.
   * pgvector supports cosine, l2, and inner_product. pgembedding supports
   * cosine, l2, and manhattan.
   * @param metric - The metric to validate.
   */
  private validateSelectedMetric(metric: Metric) {
    if (!this.allowedMetrics().includes(metric)) {
      throw new Error(
        `Invalid metric: ${metric}. Allowed metrics are: ${this.allowedMetrics().join(
          ", "
        )}`
      );
    }
  }

  /**
   * Build the SQL statement to fetch rows from the database.
   * @param returns - The columns to return from the database.
   * @param tableName - The name of the table to fetch rows from.
   * @returns {string} The SQL statement to fetch rows from the database.
   */
  abstract buildFetchRowsStatement(
    vector: number[],
    vectorColumnName: string,
    tableName: string,
    returns: string[],
    disambiguate?: boolean
  ): Fragment;

  /**
   * Build the SQL statement to ensure the extension is installed in the
   * database.
   */
  abstract buildEnsureExtensionStatement(): string;

  /**
   * Build the SQL statement to get the data type of the embedding column
   * the chosen embedding extension uses.
   */
  abstract buildDataTypeStatement(): string;

  /**
   * Build the SQL statement to insert a vector into the database.
   * @param vector - The vector to insert into the database.
   */
  abstract buildInsertVectorStatement(vector: number[]): string;

  /**
   * Build the SQL statement to create an HNSW index on the embedding column.
   * pgvector (post v0.5.0) and pgembedding both support HNSW indexes, but it's
   * still the simplest form of indexing (i.e. no partitioned table index, partial
   * index, etc etc.)
   * @param tableName - The name of the table to create the index on.
   * @param indexOpts - Options for the index.
   */
  abstract buildHNSWIndexStatement(
    indexName: string,
    tableName: string,
    columnName: string,
    indexOpts: { m?: number; efConstruction?: number; efSearch?: number }
  ): string;

  abstract runQueryWrapper<R>(
    dbInstance: IBaseProtocol<TableShape<Columns>>,
    query: PGQuery<R>
  ): Promise<R>;
}

export class PGEmbeddingExt<
  Columns extends Record<string, unknown> = Record<string, unknown>
> extends PostgresEmbeddingExtension<Columns> {
  allowedMetrics(): Metric[] {
    return ["cosine", "l2", "manhattan"];
  }

  buildFetchRowsStatement(
    vector: number[],
    vectorColumnName: string,
    tableName: string,
    returns: string[],
    disambiguate?: boolean
  ): Fragment {
    let arrow;
    switch (this.selectedMetric) {
      case "cosine":
        arrow = "<=>";
        break;
      case "l2":
        arrow = "<->";
        break;
      case "manhattan":
        arrow = "<~>";
        break;
      default:
        throw new Error("Invalid metric");
    }
    let selectStatement;
    if (disambiguate) {
      selectStatement =
        returns.length > 0
          ? returns.map((col) => `${tableName}.${col} AS ${col}`).join(", ")
          : "*";
    } else {
      selectStatement = returns.length > 0 ? returns.join(", ") : "*";
    }

    return new Fragment(
      `SELECT ${selectStatement}, $1:name ${arrow} $2:value AS "_distance" FROM $3:name`,
      [vectorColumnName, `array[${vector.join(",")}]`, tableName]
    );
  }

  buildEnsureExtensionStatement(): string {
    return "CREATE EXTENSION IF NOT EXISTS embedding;";
  }

  buildDataTypeStatement(): string {
    return "REAL[]";
  }

  buildInsertVectorStatement(vector: number[]): string {
    return `{${vector.join(",")}}`;
  }

  buildHNSWIndexStatement(
    indexName: string,
    tableName: string,
    columnName: string,
    {
      m = 16,
      efConstruction = 64,
      efSearch,
    }: { m?: number; efConstruction?: number; efSearch?: number }
  ): string {
    const opts = [`dims = ${this.dims}`];
    if (m) opts.push(`m = ${m}`);
    if (efConstruction) opts.push(`efconstruction = ${efConstruction}`);
    if (efSearch) opts.push(`efsearch = ${efSearch}`);

    let ops;
    switch (this.selectedMetric) {
      case "cosine":
        ops = " ann_cos_ops";
        break;
      case "l2":
        ops = "";
        break;
      case "manhattan":
        ops = " ann_manhattan_ops";
        break;
      default:
        throw new Error("Invalid metric");
    }
    return `CREATE INDEX IF NOT EXISTS ${indexName} ON ${tableName} USING hnsw(${columnName}${ops}) WITH (${opts.join(
      ", "
    )});`;
  }

  /**
   * For some reason, pgembedding requires you to set enable_seqscan to off
   * when using its with hnsw indexes. This function wraps the query with a
   * transaction that sets enable_seqscan to off.
   * @param PGInstance
   * @param query
   * @returns
   */
  runQueryWrapper<R>(
    dbInstance: IBaseProtocol<TableShape<Columns>>,
    query: PGQuery<R>
  ): Promise<R> {
    return dbInstance.tx(async (t) => {
      await t.none("SET LOCAL enable_seqscan = off;");
      return query(t);
    });
  }
}

export class PGVectorExt<
  Columns extends Record<string, unknown> = Record<string, unknown>
> extends PostgresEmbeddingExtension<Columns> {
  allowedMetrics(): Metric[] {
    return ["cosine", "l2", "inner_product"];
  }

  buildFetchRowsStatement(
    vector: number[],
    vectorColumnName: string,
    tableName: string,
    returns: string[],
    disambiguate?: boolean
  ): Fragment {
    let embeddingStatement;
    switch (this.selectedMetric) {
      case "cosine":
        embeddingStatement = `1 - ($1:name <=> $2::vector)`;
        break;
      case "l2":
        embeddingStatement = `$1:name <-> $2::vector`;
        break;
      case "inner_product":
        embeddingStatement = `($1:name <#> $2::vector) * -1`;
        break;
      default:
        throw new Error("Invalid metric");
    }
    let selectStatement;
    if (disambiguate) {
      selectStatement =
        returns.length > 0
          ? returns.map((col) => `${tableName}.${col} AS ${col}`).join(", ")
          : "*";
    } else {
      selectStatement = returns.length > 0 ? returns.join(", ") : "*";
    }
    const queryStr = `[${vector.join(",")}]`;
    return new Fragment(
      `SELECT ${selectStatement}, ${embeddingStatement} AS "_distance" FROM $3:name`,
      [vectorColumnName, queryStr, tableName]
    );
  }

  buildEnsureExtensionStatement(): string {
    return "CREATE EXTENSION IF NOT EXISTS vector;";
  }

  buildDataTypeStatement(): string {
    return `vector(${this.dims})`;
  }

  buildInsertVectorStatement(vector: number[]): string {
    return `[${vector.join(",")}]`;
  }

  buildHNSWIndexStatement(
    indexName: string,
    tableName: string,
    columnName: string,
    {
      m = 16,
      efConstruction = 64,
      efSearch,
    }: { m?: number; efConstruction?: number; efSearch?: number }
  ): string {
    let efSearchStr = "";
    const opts = [];
    if (m) opts.push(`m = ${m}`);
    if (efConstruction) opts.push(`ef_construction = ${efConstruction}`);
    if (efSearch) {
      efSearchStr = `\nSET hnsw.ef_search = ${efSearch};`;
    }

    let ops;
    switch (this.selectedMetric) {
      case "cosine":
        ops = " vector_cosine_ops";
        break;
      case "l2":
        ops = " vector_l2_ops";
        break;
      case "inner_product":
        ops = " vector_ip_ops";
        break;
      default:
        throw new Error("Invalid metric");
    }
    return `CREATE INDEX IF NOT EXISTS ${indexName} ON ${tableName} USING hnsw(${columnName}${ops})${
      opts.length > 0 ? ` WITH (${opts.join(", ")})` : ""
    };${efSearchStr}`;
  }

  runQueryWrapper<R>(
    dbInstance: IBaseProtocol<TableShape<Columns>>,
    query: PGQuery<R>
  ): Promise<R> {
    return query(dbInstance);
  }
}

/**
 * A column in the database. Used for extra columns.
 */
export type Column = {
  type: string;
  name: string;
  returned: boolean;
  notNull?: boolean;
  references?:
    | string
    | {
        table: string;
        column: string;
      };
};

export interface PGVectorStoreArgs<
  T extends Record<string, unknown> = Record<string, unknown>
> {
  postgresConnectionOptions:
    | IBaseProtocol<TableShape<T>>
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    | Record<string, any>
    | string;
  useHnswIndex: boolean;
  tableName?: string;
  pgExtensionOpts?:
    | { type: "pgvector" | "pgembedding"; dims?: number; metric?: Metric }
    | PostgresEmbeddingExtension;
  columns?: {
    idColumnName?: string;
    vectorColumnName?: string;
    contentColumnName?: string;
    metadataColumnName?: string;
  };
  extraColumns?: Column[];
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function isIDatabase(obj: any): obj is IDatabase<any> {
  // Here, define what properties you expect IDatabase to have.
  // For example, assume that IDatabase should have a 'one' method:
  return obj && typeof obj.one === "function";
}

export class PGVectorStore<
  T extends Record<string, unknown> = Record<string, unknown>
> extends VectorStore {
  declare FilterType: PGFilterWithJoin | Record<string, unknown>;

  pgInstance: IDatabase<TableShape<T>>;

  pgExtension: PostgresEmbeddingExtension;

  useHnswIndex: boolean;

  private extraColumns: Column[];

  private tableName: string;

  private pageContentColumnName: string;

  private vectorColumnName: string;

  private metadataColumnName: string;

  private idColumnName: string;

  constructor(embeddings: Embeddings, args: PGVectorStoreArgs) {
    super(embeddings, args);
    this.embeddings = embeddings;
    this.useHnswIndex = args.useHnswIndex;
    this.tableName = args.tableName ?? "documents";

    this.pageContentColumnName = args.columns?.contentColumnName ?? "content";
    this.vectorColumnName = args.columns?.vectorColumnName ?? "embedding";
    this.idColumnName = args.columns?.idColumnName ?? "id";
    this.metadataColumnName = args.columns?.metadataColumnName ?? "metadata";

    this.extraColumns = args.extraColumns ?? [];

    if (
      typeof args.postgresConnectionOptions === "string" ||
      (typeof args.postgresConnectionOptions === "object" &&
        ("host" in args.postgresConnectionOptions ||
          "database" in args.postgresConnectionOptions))
    ) {
      this.pgInstance = pgp(args.postgresConnectionOptions) as IDatabase<
        TableShape<T>
      >;
    } else if (isIDatabase(args.postgresConnectionOptions)) {
      this.pgInstance = args.postgresConnectionOptions as IDatabase<
        TableShape<T>
      >;
    } else {
      throw new Error("Invalid pg-promise argument");
    }

    if (
      typeof args.pgExtensionOpts === "object" &&
      "type" in args.pgExtensionOpts
    ) {
      if (args.pgExtensionOpts.type === "pgvector") {
        this.pgExtension = new PGVectorExt({
          pgDb: this.pgInstance,
          dims: args.pgExtensionOpts.dims,
          metric: args.pgExtensionOpts.metric,
        });
      } else if (args.pgExtensionOpts.type === "pgembedding") {
        this.pgExtension = new PGEmbeddingExt({
          pgDb: this.pgInstance,
          dims: args.pgExtensionOpts.dims,
          metric: args.pgExtensionOpts.metric,
        });
      }
    } else if (
      args.pgExtensionOpts != null &&
      // eslint-disable-next-line no-instanceof/no-instanceof
      args.pgExtensionOpts instanceof PostgresEmbeddingExtension
    ) {
      this.pgExtension = args.pgExtensionOpts;
    } else {
      this.pgExtension = new PGVectorExt({ pgDb: this.pgInstance });
    }
  }

  /**
   * Static method to create a new `PGVectorStore` instance from a
   * connection. It creates a table if one does not exist, and calls
   * `connect` to return a new instance of `PGVectorStore`.
   *
   * @param embeddings - Embeddings instance.
   * @param fields - `PGVectorStoreArgs` instance.
   * @returns A new instance of `PGVectorStore`.
   */
  static async initialize(
    embeddings: Embeddings,
    config: PGVectorStoreArgs
  ): Promise<PGVectorStore> {
    const postgresqlVectorStore = new PGVectorStore(embeddings, config);

    await postgresqlVectorStore._initializeClient();
    await postgresqlVectorStore.ensureTableInDatabase();

    return postgresqlVectorStore;
  }

  protected async _initializeClient() {
    await this.pgInstance.connect();
  }

  _vectorstoreType(): string {
    return "pg";
  }

  /**
   * Functions that executes the SQL queries. Can be used to modify how the queries
   * are being executed by inheriting classes. Very very useful when combined
   * with, let's say, transactions to leverage Postgres's RLS feature
   * @param query
   * @returns { PGT.QueryBuilder | PGT.Raw }
   */
  protected runQuery<R>(query: PGQuery<R>) {
    if (this.useHnswIndex) {
      return this.pgExtension.runQueryWrapper(this.pgInstance, query);
    } else {
      return query(this.pgInstance);
    }
  }

  async addVectors(
    vectors: number[][],
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    documents: Document<Record<string, any>>[],
    options?: {
      extraColumns?: (ColumnValue | null)[];
      ids?: string[];
    }
  ): Promise<void> {
    const chunkSize = 1000;
    for (let i = 0; i < vectors.length; i += chunkSize) {
      const chunkVectors = vectors.slice(i, i + chunkSize);
      const chunkDocuments = documents.slice(i, i + chunkSize);
      const chunkExtraColumns =
        options?.extraColumns?.slice(i, i + chunkSize) || [];

      const rows = chunkVectors.map((embedding, idx) => {
        const extraColumns = Object.entries(
          chunkExtraColumns[idx] ?? {}
        ).reduce((acc, [key, value]) => {
          // check if key is in this.extraColumns
          const column = this.extraColumns.find(({ name }) => name === key);
          if (column == null) return acc;

          acc[key] = value;
          return acc;
        }, {} as ColumnValue);

        const embeddingString =
          this.pgExtension.buildInsertVectorStatement(embedding);
        const documentRow = {
          [this.pageContentColumnName]: chunkDocuments[idx].pageContent,
          [this.vectorColumnName]: embeddingString,
          [this.metadataColumnName]: chunkDocuments[idx].metadata,
          ...extraColumns,
        };

        if (options?.ids?.[idx]) {
          documentRow[this.idColumnName] = options.ids[idx];
        }

        return documentRow;
      });

      await this.runQuery((database) => {
        const exclCols = Object.keys(rows[0])
          .map((key) => `${key} = EXCLUDED.${key}`)
          .join(", ");

        const columnSet = new pgp.helpers.ColumnSet(Object.keys(rows[0]), {
          table: this.tableName,
        });

        const insertQuery = `${pgp.helpers.insert(
          rows,
          columnSet
        )} ON CONFLICT (${this.idColumnName}) DO UPDATE SET ${exclCols}`;
        return database.none(insertQuery);
      });
    }
  }

  async addDocuments(
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    documents: Document<Record<string, any>>[],
    options?: {
      extraColumns: (ColumnValue | null)[];
    }
  ): Promise<void | string[]> {
    const texts = documents.map(({ pageContent }) => pageContent);
    return this.addVectors(
      await this.embeddings.embedDocuments(texts),
      documents,
      options
    );
  }

  async ensureTableInDatabase(): Promise<void> {
    await this.pgInstance.none(
      this.pgExtension.buildEnsureExtensionStatement()
    );
    await this.pgInstance.none('CREATE EXTENSION IF NOT EXISTS "uuid-ossp";');

    const columns = [
      `$2:name uuid NOT NULL DEFAULT uuid_generate_v4() PRIMARY KEY`,
      `$3:name text`,
      `$4:name jsonb`,
      `$5:name ${this.pgExtension.buildDataTypeStatement()}`,
    ];

    const params = [
      this.idColumnName,
      this.pageContentColumnName,
      this.metadataColumnName,
      this.vectorColumnName,
    ];
    let parameterCount = 5;

    const extraColumns = [];

    for (let i = 0; i < this.extraColumns.length; i += 1) {
      const { name, type, notNull, references } = this.extraColumns[i];
      let refString = "";
      const refParams = [];
      let adder = 2;
      if (references) {
        if (typeof references === "string") {
          refString = ` REFERENCES $${parameterCount + 3}:name`;
          refParams.push(references);
          adder = 3;
        } else {
          refString = ` REFERENCES $${parameterCount + 3}:name ($${
            parameterCount + 4
          }:name)`;
          refParams.push(references.table);
          refParams.push(references.column);
          adder = 4;
        }
      }

      extraColumns.push(
        `$${parameterCount + 1}:name $${parameterCount + 2}:alias${
          notNull ? " NOT NULL" : ""
        }${refString}`
      );
      params.push(name);
      params.push(type);
      params.push(...refParams);
      parameterCount += adder;
    }

    await this.pgInstance.none(
      `
      CREATE TABLE IF NOT EXISTS $1:name (${[...columns, ...extraColumns].join(
        ", "
      )});
    `,
      [this.tableName, ...params]
    );
  }

  private buildJoinStatement(statement: JoinStatement): Fragment {
    const { op, table, on } = statement;
    if (
      ![
        "JOIN",
        "LEFT JOIN",
        "RIGHT JOIN",
        "FULL JOIN",
        "CROSS JOIN",
        "INNER JOIN",
      ].includes(op)
    ) {
      throw new Error(`Invalid join statement: ${op}`);
    }

    const parametrizedOn = [];
    const onParams = [];
    for (let k = 0; k < on.length; k += 2) {
      // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
      if (on[k].operator && !allowedOperators.includes(on[k].operator!)) {
        throw new Error(`Invalid operator: ${on[k].operator}`);
      }
      parametrizedOn.push(
        `$${k + 2}:alias ${on[k].operator || "="} $${k + 3}:alias`
      );
      onParams.push(on[k].left);
      onParams.push(on[k].right);
    }

    return new Fragment(`${op} $1:name ON ${parametrizedOn.join(" AND ")}`, [
      table,
      ...onParams,
    ]);
  }

  private async fetchRows(
    query: number[],
    k: number,
    filter?: this["FilterType"],
    returnedColumns?: string[]
  ) {
    let filterStatement: Fragment | null = null;
    let shouldDisambiguate = false;

    let joinStatements: Fragment[] = [];

    if (filter) {
      const { metadataFilter, columnFilter, join, ...rest } = filter;
      if (metadataFilter || columnFilter) {
        const metadataFilter = filter.metadataFilter as PGFilter | undefined;
        const columnFilter = filter.columnFilter as PGFilter | undefined;
        let filterType: "metadata" | "column" | undefined;

        if (metadataFilter && columnFilter) {
          throw new Error("Cannot have both metadataFilter and columnFilter");
        } else if (metadataFilter) {
          filterType = "metadata";
        } else if (columnFilter) {
          filterType = "column";
        }
        const combinedFilters = this.buildSqlFilterStr(
          metadataFilter ?? columnFilter,
          filterType
        );
        if (combinedFilters) {
          filterStatement = Fragment.combineFragments(combinedFilters);
        }
      } else if (rest && Object.keys(rest).length > 0) {
        filterStatement = new Fragment(`WHERE $1:name @> $2`, [
          this.metadataColumnName,
          rest,
        ]);
      }

      if (join) {
        const joinStatement = filter.join as
          | JoinStatement
          | JoinStatement[]
          | undefined;
        if (Array.isArray(joinStatement)) {
          joinStatements = joinStatement.map((statement) =>
            this.buildJoinStatement(statement)
          );
        } else {
          joinStatements = joinStatement
            ? [this.buildJoinStatement(joinStatement)]
            : [];
        }
        if (joinStatements.length > 0) {
          shouldDisambiguate = true;
        }
      }
    }

    const selectedColumns = [
      ...(returnedColumns ?? []),
      ...this.extraColumns
        .filter(({ returned }) => returned)
        .map(({ name }) => name),
    ];
    const queryFrags = [
      this.pgExtension.buildFetchRowsStatement(
        query,
        this.vectorColumnName,
        this.tableName,
        [
          this.idColumnName,
          this.pageContentColumnName,
          this.metadataColumnName,
          ...selectedColumns,
        ],
        shouldDisambiguate
      ),
      ...joinStatements,
      filterStatement,
      new Fragment(`ORDER BY "_distance" LIMIT $1;`, [k]),
    ];

    const combinedFrags = Fragment.combineFragments(queryFrags);

    const results = await this.runQuery((database) =>
      database.any(combinedFrags.query, combinedFrags.values)
    );

    if (results && results.length > 0 && this.vectorColumnName in results[0]) {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      results.forEach((_: any, idx: number) => {
        if (typeof results[idx][this.vectorColumnName] === "string") {
          try {
            results[idx][this.vectorColumnName] = JSON.parse(
              results[idx][this.vectorColumnName]
            );
          } catch (error) {
            throw new Error("Error parsing embedding");
          }
        }
      });
    }

    return results;
  }

  async similaritySearchVectorWithScore(
    query: number[],
    k: number,
    filter?: this["FilterType"] | undefined
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
  ): Promise<[Document<Record<string, any>>, number][]> {
    const rows = await this.fetchRows(query, k, filter);
    return rows.map((row) => [
      new Document({
        pageContent: row[this.pageContentColumnName] as string,
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        metadata: row[this.metadataColumnName] as Record<string, any>,
      }),
      row._distance,
    ]);
  }

  async maxMarginalRelevanceSearch(
    query: string,
    options: MaxMarginalRelevanceSearchOptions<this["FilterType"]>
  ): Promise<Document[]> {
    const { k, fetchK = 20, lambda = 0.7, filter } = options;
    const queryEmbedding = await this.embeddings.embedQuery(query);
    const results = await this.fetchRows(queryEmbedding, fetchK, filter, [
      this.vectorColumnName,
    ]);

    const embeddings = results.map((result) => result[this.vectorColumnName]);
    const mmrIndexes = maximalMarginalRelevance(
      queryEmbedding,
      embeddings,
      lambda,
      k
    );
    return mmrIndexes
      .filter((idx) => idx !== -1)
      .map((idx) => {
        const result = results[idx];
        return new Document({
          pageContent: result[this.pageContentColumnName] as string,
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          metadata: result[this.metadataColumnName] as Record<string, any>,
        });
      });
  }

  /**
   * Build the HNSW index on the embedding column. Should be
   * called before doing similarity search if you plan to have
   * a large number of rows in the table.
   * @param buildIndexOpt
   */
  async buildIndex(
    indexName: string,
    {
      m,
      efConstruction,
      efSearch,
    }: {
      m?: number;
      efConstruction?: number;
      efSearch?: number;
    } = {}
  ): Promise<void> {
    await this.pgInstance.none(
      this.pgExtension.buildHNSWIndexStatement(
        indexName,
        this.tableName,
        this.vectorColumnName,
        {
          m,
          efConstruction,
          efSearch,
        }
      )
    );
  }

  async dropIndex(indexName: string): Promise<void> {
    await this.pgInstance.none(`DROP INDEX IF EXISTS ${indexName};`);
  }

  buildTextSearchStatement(
    key: string,
    param: TextSearchValue,
    column: string
  ) {
    const { query, type, config } = param;
    const lang = `${config ?? "simple"}`;
    let queryOp = "to_tsquery";
    if (type) {
      switch (type) {
        case "plain":
          queryOp = "plainto_tsquery";
          break;
        case "phrase":
          queryOp = "phraseto_tsquery";
          break;
        case "websearch":
          queryOp = "websearch_to_tsquery";
          break;
        default:
          throw new Error("Invalid text search type");
      }
    }

    return new Fragment(`to_tsvector($1, ${column}) @@ ${queryOp}($3, $4)`, [
      lang,
      key,
      lang,
      query,
    ]);
  }

  private buildColumnStatement(
    value: string | number,
    type: string,
    index: number
  ) {
    let columnStatement;
    let typeCast: string;
    let arrow = "";

    if (isString(value)) {
      typeCast = "::text";
      arrow = "->>";
    } else if (isInt(value)) {
      typeCast = "::int";
      arrow = "->";
    } else if (isFloat(value)) {
      typeCast = "::float";
      arrow = "->";
    } else {
      throw new Error("Data type not supported");
    }

    if (type === "column") {
      columnStatement = `$${index}:alias`;
    } else {
      columnStatement = `(${this.metadataColumnName}${arrow}$${index})${typeCast}`;
    }

    return columnStatement;
  }

  /**
   * Build the SQL filter string from the filter object.
   * @param filter - The filter object
   * @returns
   */
  buildSqlFilterStr(
    filter?: PGFilter,
    type: "metadata" | "column" = "metadata"
  ) {
    if (filter == null) return null;

    const buildClause = (
      key: string,
      operator: ComparisonMapKey,
      value: string | number | TextSearchValue
    ): Fragment => {
      const op = operator;
      const compRaw = ComparisonMap[op];
      const myValue =
        typeof value === "object" && "query" in value ? value.query : value;

      if (op === "$textSearch") {
        const columnStatement = this.buildColumnStatement(myValue, type, 2);
        return this.buildTextSearchStatement(
          key,
          value as TextSearchValue,
          columnStatement
        );
      } else {
        const columnStatement = this.buildColumnStatement(myValue, type, 1);
        return new Fragment(`${columnStatement} ${compRaw} $2`, [key, myValue]);
      }
    };
    const allowedOps = Object.keys(LogicalMap);

    const recursiveBuild = (filterObj: PGFilter): Fragment[] => {
      const fragmentList: Fragment[] = [];

      Object.entries(filterObj).forEach(([key, ops]) => {
        if (allowedOps.includes(key)) {
          const logicalParts = (ops as PGFilter[]).flatMap(recursiveBuild);
          const separator = LogicalMap[key as LogicalMapKey];
          const combinedQuery = Fragment.combineFragments(
            logicalParts,
            ` ${separator} `,
            true
          );
          fragmentList.push(combinedQuery);
          return;
        }

        if (typeof ops === "object" && !Array.isArray(ops)) {
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          Object.entries(ops as Record<string, any>).forEach(
            ([opName, value]) => {
              if (value) {
                const clause = buildClause(
                  key,
                  opName as ComparisonMapKey,
                  value
                );
                fragmentList.push(clause);
              }
            }
          );
          return;
        }

        const clause = buildClause(key, "$eq", ops);
        fragmentList.push(clause);
      });

      return fragmentList;
    };

    const built = Fragment.combineFragments(
      recursiveBuild(filter),
      ` AND `,
      true
    );

    if (built.query.length === 0) return null;
    return ["WHERE", built];
  }

  /**
   * Static method to create a new `PGVectorStore` instance from an
   * array of `Document` instances. It adds the documents to the store.
   *
   * @param docs - Array of `Document` instances.
   * @param embeddings - Embeddings instance.
   * @param dbConfig - `PGVectorStoreArgs` instance.
   * @returns Promise that resolves with a new instance of `PGVectorStore`.
   */
  static async fromDocuments(
    docs: Document[],
    embeddings: Embeddings,
    dbConfig: PGVectorStoreArgs
  ): Promise<PGVectorStore> {
    const instance = await PGVectorStore.initialize(embeddings, dbConfig);
    await instance.addDocuments(docs);

    return instance;
  }

  /**
   * Closes all the clients in the pool and terminates the pool.
   *
   * @returns Promise that resolves when all clients are closed and the pool is terminated.
   */
  async end(): Promise<void> {
    return this.pgInstance.$pool.end();
  }
}
