import type { Knex as KnexT } from "knex";
import { MaxMarginalRelevanceSearchOptions, VectorStore } from "./base.js";
import { Embeddings } from "../embeddings/base.js";
import { Document } from "../document.js";
import { maximalMarginalRelevance } from "../util/math.js";
import { isFloat, isInt, isString } from "../util/types.js";

export type FilterValue = string | number;

export type TextSearchValue = {
  query: string;
  type?: "plain" | "phrase" | "websearch";
  config?: string;
};

export type ComparisonOperator =
  | { $eq: FilterValue }
  | { $gt: FilterValue }
  | { $gte: FilterValue }
  | { $lt: FilterValue }
  | { $lte: FilterValue }
  | { $not: FilterValue }
  | { $textSearch: TextSearchValue };

export type LogicalOperator = { $and: KnexFilter[] } | { $or: KnexFilter[] };

export type ExcludeKeyValueFilter = "$filter" | "$join";

export type KeyValueFilter = {
  [key: string]: FilterValue | ComparisonOperator;
} & {
  [key in ExcludeKeyValueFilter]?: never;
};

export type KnexFilter = KeyValueFilter | LogicalOperator;

export type KnexFilterWithJoin =
  | {
      metadataFilter?: never;
      columnFilter?: KnexFilter;
      join?: string;
    }
  | {
      metadataFilter?: KnexFilter;
      columnFilter?: never;
      join?: string;
    };

export type Metric = "cosine" | "l2" | "manhattan" | "inner_product";

export type ExtensionOpts = {
  /**
   * The metric to use for similarity search.
   */
  metric?: Metric;

  /**
   * The number of dimensions of the embeddings.
   */
  dims?: number;
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
export abstract class PostgresEmbeddingExtension {
  abstract allowedMetrics: Metric[];

  selectedMetric: Metric;

  /**
   * The number of dimensions of the embeddings.
   */
  dims: number;

  constructor(extensionOpts?: ExtensionOpts) {
    const metric = extensionOpts?.metric ?? "cosine";
    this.validateSelectedMetric(metric);
    this.selectedMetric = metric;
    this.dims = extensionOpts?.dims ?? 1536; // defaults to OpenAI 1536 embedding dims
  }

  /**
   * Validate the selected metric. Check if it is one of the allowed metrics.
   * pgvector supports cosine, l2, and inner_product. pgembedding supports
   * cosine, l2, and manhattan.
   * @param metric - The metric to validate.
   */
  private validateSelectedMetric(metric: Metric) {
    if (!this.allowedMetrics.includes(metric)) {
      throw new Error(
        `Invalid metric: ${metric}. Allowed metrics are: ${this.allowedMetrics.join(
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
    returns: string[],
    tableName: string
  ): string;

  /**
   * Build the SQL statement to ensure the extension is installed in the
   * database.
   */
  abstract buildEnsureExtensionStatement(): string;

  /**
   * Build the SQL statement to get the data type of the embedding column
   * the chosen embedding extension uses.
   */
  abstract buildDataType(): string;

  /**
   * Build the SQL statement to insert a vector into the database.
   * @param vector - The vector to insert into the database.
   */
  abstract buildInsertionVector(vector: number[]): string;

  /**
   * Build the SQL statement to create an HNSW index on the embedding column.
   * @param tableName - The name of the table to create the index on.
   * @param indexOpts - Options for the index.
   */
  abstract buildHNSWIndexStatement(
    tableName: string,
    columnName: string,
    indexOpts: { m?: number; efConstruction?: number; efSearch?: number }
  ): string;
}

export class PGEmbeddingExt extends PostgresEmbeddingExtension {
  allowedMetrics: Metric[] = ["cosine", "l2", "manhattan"];

  buildFetchRowsStatement(returns: string[], tableName: string): string {
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

    const selectStatement = returns.length > 0 ? returns.join(", ") : "*";

    return `SELECT ${selectStatement}, embedding ${arrow} array? AS "_distance" FROM ${tableName}`;
  }

  buildEnsureExtensionStatement(): string {
    return "CREATE EXTENSION IF NOT EXISTS embedding;";
  }

  buildDataType(): string {
    return "REAL[]";
  }

  buildInsertionVector(vector: number[]): string {
    return `{${vector.join(",")}}`;
  }

  buildHNSWIndexStatement(
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
    return `CREATE INDEX ON ${tableName} USING hnsw(${columnName}${ops}) WITH (${opts.join(
      ", "
    )});`;
  }
}

export class PGVectorExt extends PostgresEmbeddingExtension {
  allowedMetrics: Metric[] = ["cosine", "l2", "inner_product"];

  buildFetchRowsStatement(returns: string[], tableName: string): string {
    let embeddingStatement;
    switch (this.selectedMetric) {
      case "cosine":
        embeddingStatement = "1 - (embedding <=> ?::vector)";
        break;
      case "l2":
        embeddingStatement = "embedding <-> ?::vector";
        break;
      case "inner_product":
        embeddingStatement = "(embedding <#> ?::vector) * -1";
        break;
      default:
        throw new Error("Invalid metric");
    }

    const selectStatement = returns.length > 0 ? returns.join(", ") : "*";

    return `SELECT ${selectStatement}, ${embeddingStatement} AS "_distance" FROM ${tableName}`;
  }

  buildEnsureExtensionStatement(): string {
    return "CREATE EXTENSION IF NOT EXISTS vector;";
  }

  buildDataType(): string {
    return "vector";
  }

  buildInsertionVector(vector: number[]): string {
    return `[${vector.join(",")}]`;
  }

  buildHNSWIndexStatement(
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
    return `CREATE INDEX ON ${tableName} USING hnsw(${columnName}${ops})${
      opts.length > 0 ? ` WITH (${opts.join(", ")})` : ""
    };${efSearchStr}`;
  }
}

/**
 * A search result returned by the KnexVectorStore.
 */
export type SearchResult = {
  pageContent: string;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  metadata: Record<string, any>;
  embedding: number[];
  _distance: number;
};

/**
 * A column in the database. Used for extra columns.
 */
export type Column = {
  type: string;
  name: string;
  returned: boolean;
};

export interface KnexVectorStoreArgs {
  knex: KnexT;
  tableName?: string;
  pageContentColumn?: string;
  pgExtension?: "pgvector" | "pgembedding" | PostgresEmbeddingExtension;
  extraColumns?: Column[];
}

export class KnexVectorStore extends VectorStore {
  declare FilterType: KnexFilterWithJoin;

  knex: KnexT;

  tableName: string;

  pageContentColumn: string;

  pgExtension: PostgresEmbeddingExtension;

  extraColumns: Column[];

  constructor(embeddings: Embeddings, args: KnexVectorStoreArgs) {
    super(embeddings, args);
    this.embeddings = embeddings;
    this.knex = args.knex;
    this.tableName = args.tableName ?? "documents";
    this.pageContentColumn = args.pageContentColumn ?? "content";
    this.extraColumns = args.extraColumns ?? [];

    if (args.pgExtension === "pgvector") {
      this.pgExtension = new PGVectorExt();
    } else if (args.pgExtension === "pgembedding") {
      this.pgExtension = new PGEmbeddingExt();
    } else if (
      args.pgExtension != null &&
      // eslint-disable-next-line no-instanceof/no-instanceof
      args.pgExtension instanceof PostgresEmbeddingExtension
    ) {
      this.pgExtension = args.pgExtension;
    } else {
      this.pgExtension = new PGVectorExt();
    }
  }

  _vectorstoreType(): string {
    return "knex";
  }

  /**
   * Functions that executes the SQL queries. Can be used to modify how the queries
   * are being executed by inheriting classes. Very very useful when combined
   * with, let's say, transactions, by doing something like
   * protected doQuery(query: (db: KnexT) => KnexT.QueryBuilder | KnexT.Raw) {
   *   return this.knex.transaction((trx) => {
   *     return trx
   *       .raw(`SELECT set_config(?, ?, true)`, someValueA, someValueB)
   *       .then(() => {
   *         return query(trx);
   *       });
   *   });
   * }
   * to leverage Postgres's RLS feature
   * @param query
   * @returns { KnexT.QueryBuilder | KnexT.Raw }
   */
  protected doQuery(query: (db: KnexT) => KnexT.QueryBuilder | KnexT.Raw) {
    return query(this.knex);
  }

  async addVectors(
    vectors: number[][],
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    documents: Document<Record<string, any>>[],
    options?: {
      extraColumns: (ColumnValue | null)[];
    }
  ): Promise<void> {
    await this.ensureTableInDatabase();
    const rows = vectors.map((embedding, idx) => {
      const extraColumns = Object.entries(
        options?.extraColumns[idx] ?? {}
      ).reduce((acc, [key, value]) => {
        // check if key is in this.extraColumns
        const column = this.extraColumns.find(({ name }) => name === key);
        if (column == null) return acc;

        acc[key] = value;
        return acc;
      }, {} as ColumnValue);

      const embeddingString = this.pgExtension.buildInsertionVector(embedding);
      const documentRow = {
        [this.pageContentColumn]: documents[idx].pageContent,
        embedding: embeddingString,
        metadata: documents[idx].metadata,
        ...extraColumns,
      };

      return documentRow;
    });
    await this.doQuery((database) => database(this.tableName).insert(rows));
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
    await this.knex.raw(this.pgExtension.buildEnsureExtensionStatement());
    await this.knex.raw('CREATE EXTENSION IF NOT EXISTS "uuid-ossp";');

    const columns = [
      '"id" uuid NOT NULL DEFAULT uuid_generate_v4() PRIMARY KEY',
      `"${this.pageContentColumn}" text`,
      '"metadata" jsonb',
      `"embedding" ${this.pgExtension.buildDataType()}`,
    ];

    const extraColumns = this.extraColumns.map(
      ({ name, type }) => `"${name}" ${type}`
    );

    await this.knex.raw(`
      CREATE TABLE IF NOT EXISTS ${this.tableName} (
        ${[...columns, ...extraColumns].join("\n,")}
      );
    `);
  }

  private async fetchRows(
    query: number[],
    k: number,
    filter?: this["FilterType"] | undefined,
    returnedColumns?: string[]
  ): Promise<SearchResult[]> {
    const { metadataFilter, columnFilter, join: joinStatement } = filter ?? {};
    let filterType: "metadata" | "column" | undefined;

    if (metadataFilter && columnFilter) {
      throw new Error("Cannot have both metadataFilter and columnFilter");
    } else if (metadataFilter) {
      filterType = "metadata";
    } else if (columnFilter) {
      filterType = "column";
    }

    const vector = `[${query.join(",")}]`;
    const selectedColumns = [
      ...(returnedColumns ?? []),
      ...this.extraColumns
        .filter(({ returned }) => returned)
        .map(({ name }) => name),
    ];
    const queryStr = [
      this.knex
        .raw(
          this.pgExtension.buildFetchRowsStatement(
            ["id", this.pageContentColumn, "metadata", ...selectedColumns],
            this.tableName
          ),
          [vector]
        )
        .toString(),
      joinStatement,
      this.buildSqlFilterStr(metadataFilter ?? columnFilter, filterType),
      this.knex.raw(`ORDER BY "_distance" LIMIT ?;`, [k]).toString(),
    ]
      .filter((x) => x != null)
      .join(" ");
    const results = await this.doQuery((database) => database.raw(queryStr));
    return results.rows as SearchResult[];
  }

  async similaritySearchVectorWithScore(
    query: number[],
    k: number,
    filter?: this["FilterType"] | undefined
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
  ): Promise<[Document<Record<string, any>>, number][]> {
    const rows = await this.fetchRows(query, k, filter, ["_distance"]);
    return rows.map((row) => [
      new Document({
        pageContent: row[this.pageContentColumn as keyof typeof row] as string,
        metadata: row.metadata,
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
    const results = await this.fetchRows(queryEmbedding, fetchK, filter);

    const embeddings = results.map((result) => result.embedding);
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
          pageContent: result.pageContent,
          metadata: result.metadata,
        });
      });
  }

  /**
   * Build the HNSW index on the embedding column. Should be called after
   * adding vectors to the database, but before doing similarity search if
   * you plan to have a large number of rows in the table.
   * @param buildIndexOpt
   */
  async buildIndex({
    m,
    efConstruction,
    efSearch,
  }: {
    m?: number;
    efConstruction?: number;
    efSearch?: number;
  }): Promise<void> {
    await this.ensureTableInDatabase();
    await this.knex.raw(
      this.pgExtension.buildHNSWIndexStatement(this.tableName, "embedding", {
        m,
        efConstruction,
        efSearch,
      })
    );
  }

  buildTextSearchStatement(param: TextSearchValue, column: string) {
    const { query, type, config } = param;
    const lang = `'${config ?? "simple"}'`;
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

    return this.knex
      .raw(`to_tsvector(?, ?) @@ ${queryOp}(?, ?)`, [lang, column, lang, query])
      .toString();
  }

  /**
   * Build the SQL filter string from the filter object.
   * @param filter - The filter object
   * @returns
   */
  buildSqlFilterStr(
    filter?: KnexFilter,
    type: "metadata" | "column" = "metadata"
  ) {
    if (filter == null) return null;

    const buildClause = (
      key: string,
      operator: ComparisonMapKey,
      value: string | number | TextSearchValue
    ): string => {
      const op = operator;
      const compRaw = ComparisonMap[op];

      let typeCast;
      let arrow;

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
      let columnKey;
      if (key === this.pageContentColumn) {
        columnKey = this.pageContentColumn;
      } else {
        if (type === "column") {
          columnKey = `"${key}"`;
          typeCast = "";
        } else {
          columnKey = `metadata${arrow}"${key}"`;
        }
      }

      if (op === "$textSearch") {
        return this.buildTextSearchStatement(
          value as TextSearchValue,
          columnKey
        );
      } else {
        return this.knex
          .raw(`? ${compRaw} ?${typeCast}`, [columnKey, value])
          .toString();
      }
    };
    const allowedOps = Object.keys(LogicalMap);

    const recursiveBuild = (filterObj: KnexFilter): string =>
      Object.entries(filterObj)
        .map(([key, ops]) => {
          if (allowedOps.includes(key)) {
            const logicalParts = (ops as KnexFilter[]).map(recursiveBuild);
            const separator = LogicalMap[key as LogicalMapKey];
            return `(${logicalParts.join(` ${separator} `)})`;
          }

          if (typeof ops === "object" && !Array.isArray(ops)) {
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            return Object.entries(ops as Record<string, any>)
              .map(([opName, value]) => {
                if (!value) return null;
                return buildClause(key, opName as ComparisonMapKey, value);
              })
              .filter(Boolean)
              .join(" AND ");
          }

          return buildClause(key, "$eq", ops);
        })
        .filter(Boolean)
        .join(" AND ");

    const strFilter = `WHERE ${recursiveBuild(filter)}`;

    if (strFilter === "WHERE ") return null;
    return strFilter;
  }
}
