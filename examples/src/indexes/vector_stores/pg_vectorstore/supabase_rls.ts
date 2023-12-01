import { type IBaseProtocol } from "pg-promise";
import { PGVectorStore } from "langchain/vectorstores/pg";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";

/**
 * You can extend PGVectorStore and change how the query runs to
 * make it work with RLS. As an example, if you're using Supabase's
 * RLS feature, you can extend PGVectorStore like so:
 */

class SupabasePGVectorStore extends PGVectorStore {
  jwt?: Record<string, any>;

  setJWT(jwt: Record<string, any>) {
    this.jwt = jwt;
  }

  unsetJWT() {
    this.jwt = undefined;
  }

  protected runQuery<R>(
    query: (t: IBaseProtocol<any>) => Promise<R>
  ): Promise<R> {
    return this.pgInstance.tx((t) => {
      const claimsSetting = "request.jwt.claims";
      const claims = JSON.stringify(this.jwt);
      return (
        t
          .query(`SELECT set_config($1, $2, true);`, [claimsSetting, claims])
          /**
           * You might want to use this.pgExtension.runQueryWrapper(t, query)
           * if you're using pg_embedding, since it needs to add
           * SET LOCAL enable_seqscan = off; in the query for some reason.
           * If you're using pgvector, you can just do
           * .then(() => query(t));
           */
          .then(() => this.pgExtension.runQueryWrapper(t, query))
      );
    });
  }
}

const pgConfig = {
  database: "db",
  user: "postgres",
  password: "postgres",
  max: 20,
  host: "localhost",
  port: 5432,
};

const embedding = new OpenAIEmbeddings();
const vectorStore = new SupabasePGVectorStore(embedding, {
  postgresConnectionOptions: pgConfig,
  useHnswIndex: false,
  tableName: "pg_embeddings",
  columns: {
    contentColumnName: "content",
  },
  extraColumns: [
    {
      name: "extra_stuff",
      type: "text",
      returned: true,
      notNull: true,
    },
    {
      name: "some_table_id",
      type: "integer",
      returned: false,
      references: "some_table",
    },
  ],
  pgExtensionOpts: {
    type: "pgvector",
    dims: 1536,
    metric: "cosine",
  },
});

vectorStore.setJWT({
  "sub": "testestestest",
  "role": "user",
  "key": "somekeytest"
});
const result = await vectorStore.similaritySearch("This is a long text", 1);

console.log(result);