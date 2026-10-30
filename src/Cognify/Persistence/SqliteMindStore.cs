using Cognify.Core;
using Microsoft.Data.Sqlite;

namespace Cognify.Persistence;

/// <summary>
/// SQLite-backed persistence for a mind's complete state. One database file
/// holds one mind. Uses a simple single-row-per-collection schema with a
/// full replace-on-save strategy inside a transaction.
/// </summary>
public sealed class SqliteMindStore : IDisposable
{
    private readonly string _connectionString;
    private SqliteConnection? _connection;

    /// <summary>Opens (or creates) a mind database at the given path. Use ":memory:" for in-memory.</summary>
    public SqliteMindStore(string path)
    {
        if (string.IsNullOrWhiteSpace(path))
            throw new PersistenceException("SQLite path cannot be empty.");
        Path = path;
        _connectionString = $"Data Source={path}";
    }

    /// <summary>The database path.</summary>
    public string Path { get; }

    private SqliteConnection Connection
    {
        get
        {
            if (_connection is null)
            {
                _connection = new SqliteConnection(_connectionString);
                _connection.Open();
            }
            return _connection;
        }
    }

    /// <summary>Asynchronously saves a snapshot (full replace).</summary>
    public async Task SaveAsync(MindSnapshot snap, CancellationToken ct = default) =>
        await Task.Run(() => Save(snap), ct).ConfigureAwait(false);

    /// <summary>Asynchronously loads the snapshot from the database.</summary>
    public async Task<MindSnapshot> LoadAsync(CancellationToken ct = default) =>
        await Task.Run(Load, ct).ConfigureAwait(false);

    /// <summary>Saves a snapshot (synchronous).</summary>
    public void Save(MindSnapshot snap)
    {
        try
        {
            using var tx = Connection.BeginTransaction();
            Execute(@"
                CREATE TABLE IF NOT EXISTS profile (
                    key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS emotions (
                    emotion TEXT PRIMARY KEY, intensity REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS mood (
                    id INTEGER PRIMARY KEY CHECK (id = 1), valence REAL NOT NULL, arousal REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    kind TEXT NOT NULL, content TEXT NOT NULL, importance REAL NOT NULL,
                    strength REAL NOT NULL, created_at TEXT NOT NULL, last_recalled_at TEXT,
                    recall_count INTEGER NOT NULL, emotion_type TEXT, emotion_intensity REAL NOT NULL,
                    source TEXT);
                CREATE TABLE IF NOT EXISTS facts (
                    subject TEXT NOT NULL, predicate TEXT NOT NULL, object TEXT NOT NULL,
                    confidence REAL NOT NULL, source TEXT);
                CREATE TABLE IF NOT EXISTS rules (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS goals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS drives (
                    name TEXT PRIMARY KEY, level REAL NOT NULL, growth REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS learning (
                    id INTEGER PRIMARY KEY CHECK (id = 1), network TEXT, stats TEXT);
                CREATE TABLE IF NOT EXISTS stylenet (
                    id INTEGER PRIMARY KEY CHECK (id = 1), network TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS lexicon (
                    word TEXT PRIMARY KEY, count INTEGER NOT NULL,
                    first_cycle INTEGER NOT NULL, last_cycle INTEGER NOT NULL);
                CREATE TABLE IF NOT EXISTS lexicon_spoken (
                    word TEXT PRIMARY KEY);
                CREATE TABLE IF NOT EXISTS qtable (
                    action INTEGER PRIMARY KEY, q REAL NOT NULL, count INTEGER NOT NULL);
                CREATE TABLE IF NOT EXISTS associations (
                    node_a TEXT NOT NULL, node_b TEXT NOT NULL, weight REAL NOT NULL,
                    PRIMARY KEY (node_a, node_b));
                CREATE TABLE IF NOT EXISTS semantic (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    text TEXT NOT NULL, vector TEXT NOT NULL, tag TEXT,
                    created_at TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS sayings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, text TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS langmodel (
                    id INTEGER PRIMARY KEY CHECK (id = 1), payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS deeplangmodel (
                    id INTEGER PRIMARY KEY CHECK (id = 1), payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS selfimprovement (
                    id INTEGER PRIMARY KEY CHECK (id = 1), payload TEXT NOT NULL);", tx);

            // Full replace of dynamic tables.
            foreach (var table in new[] { "emotions", "mood", "memories", "facts", "rules", "goals", "drives", "learning", "qtable", "associations", "profile", "semantic", "sayings", "stylenet", "lexicon", "lexicon_spoken", "langmodel", "deeplangmodel", "selfimprovement" })
                Execute($"DELETE FROM {table};", tx);

            Bind("name", snap.Name, tx);
            Bind("description", snap.Description, tx);
            Bind("cycles", snap.Cycles.ToString(), tx);
            Bind("saved_at", snap.SavedAt.ToString("O"), tx);
            Bind("config", snap.ConfigJson, tx);

            foreach (var (emotion, intensity) in snap.Emotions)
            {
                using var cmd = Connection.CreateCommand();
                cmd.Transaction = tx;
                cmd.CommandText = "INSERT INTO emotions (emotion, intensity) VALUES ($e, $i);";
                cmd.Parameters.AddWithValue("$e", emotion);
                cmd.Parameters.AddWithValue("$i", intensity);
                cmd.ExecuteNonQuery();
            }

            using (var mood = Connection.CreateCommand())
            {
                mood.Transaction = tx;
                mood.CommandText = "INSERT INTO mood (id, valence, arousal) VALUES (1, $v, $a);";
                mood.Parameters.AddWithValue("$v", snap.MoodValence);
                mood.Parameters.AddWithValue("$a", snap.MoodArousal);
                mood.ExecuteNonQuery();
            }

            foreach (var m in snap.Memories)
            {
                using var cmd = Connection.CreateCommand();
                cmd.Transaction = tx;
                cmd.CommandText = @"INSERT INTO memories
                    (kind, content, importance, strength, created_at, last_recalled_at,
                     recall_count, emotion_type, emotion_intensity, source)
                    VALUES ($k, $c, $im, $st, $cr, $lr, $rc, $et, $ei, $so);";
                cmd.Parameters.AddWithValue("$k", m.Kind);
                cmd.Parameters.AddWithValue("$c", m.Content);
                cmd.Parameters.AddWithValue("$im", m.Importance);
                cmd.Parameters.AddWithValue("$st", m.Strength);
                cmd.Parameters.AddWithValue("$cr", m.CreatedAt.ToString("O"));
                cmd.Parameters.AddWithValue("$lr", (object?)m.LastRecalledAt?.ToString("O") ?? DBNull.Value);
                cmd.Parameters.AddWithValue("$rc", m.RecallCount);
                cmd.Parameters.AddWithValue("$et", (object?)m.EmotionType ?? DBNull.Value);
                cmd.Parameters.AddWithValue("$ei", m.EmotionIntensity);
                cmd.Parameters.AddWithValue("$so", (object?)m.Source ?? DBNull.Value);
                cmd.ExecuteNonQuery();
            }

            foreach (var f in snap.Facts)
            {
                using var cmd = Connection.CreateCommand();
                cmd.Transaction = tx;
                cmd.CommandText = "INSERT INTO facts (subject, predicate, object, confidence, source) VALUES ($s, $p, $o, $c, $so);";
                cmd.Parameters.AddWithValue("$s", f[0]);
                cmd.Parameters.AddWithValue("$p", f[1]);
                cmd.Parameters.AddWithValue("$o", f[2]);
                cmd.Parameters.AddWithValue("$c", f[3]);
                cmd.Parameters.AddWithValue("$so", f.Length > 4 ? f[4] : "");
                cmd.ExecuteNonQuery();
            }

            foreach (var r in snap.Rules)
            {
                using var cmd = Connection.CreateCommand();
                cmd.Transaction = tx;
                cmd.CommandText = "INSERT INTO rules (payload) VALUES ($p);";
                cmd.Parameters.AddWithValue("$p", System.Text.Json.JsonSerializer.Serialize(r));
                cmd.ExecuteNonQuery();
            }

            foreach (var g in snap.Goals)
            {
                using var cmd = Connection.CreateCommand();
                cmd.Transaction = tx;
                cmd.CommandText = "INSERT INTO goals (payload) VALUES ($p);";
                cmd.Parameters.AddWithValue("$p", System.Text.Json.JsonSerializer.Serialize(g));
                cmd.ExecuteNonQuery();
            }

            foreach (var d in snap.Drives)
            {
                using var cmd = Connection.CreateCommand();
                cmd.Transaction = tx;
                cmd.CommandText = "INSERT INTO drives (name, level, growth) VALUES ($n, $l, $g);";
                cmd.Parameters.AddWithValue("$n", d[0]);
                cmd.Parameters.AddWithValue("$l", d[1]);
                cmd.Parameters.AddWithValue("$g", d[2]);
                cmd.ExecuteNonQuery();
            }

            if (snap.NetworkJson is not null)
            {
                using var cmd = Connection.CreateCommand();
                cmd.Transaction = tx;
                cmd.CommandText = "INSERT INTO learning (id, network, stats) VALUES (1, $n, $s);";
                cmd.Parameters.AddWithValue("$n", snap.NetworkJson);
                cmd.Parameters.AddWithValue("$s", System.Text.Json.JsonSerializer.Serialize(snap.Stats));
                cmd.ExecuteNonQuery();
            }

            if (snap.StyleNetworkJson is not null)
            {
                using var cmd = Connection.CreateCommand();
                cmd.Transaction = tx;
                cmd.CommandText = "INSERT INTO stylenet (id, network) VALUES (1, $n);";
                cmd.Parameters.AddWithValue("$n", snap.StyleNetworkJson);
                cmd.ExecuteNonQuery();
            }

            foreach (var entry in snap.LexiconEntries)
            {
                using var cmd = Connection.CreateCommand();
                cmd.Transaction = tx;
                cmd.CommandText = "INSERT INTO lexicon (word, count, first_cycle, last_cycle) VALUES ($w, $c, $f, $l);";
                cmd.Parameters.AddWithValue("$w", entry[0]);
                cmd.Parameters.AddWithValue("$c", entry[1]);
                cmd.Parameters.AddWithValue("$f", entry[2]);
                cmd.Parameters.AddWithValue("$l", entry[3]);
                cmd.ExecuteNonQuery();
            }

            foreach (var word in snap.LexiconSpoken)
            {
                using var cmd = Connection.CreateCommand();
                cmd.Transaction = tx;
                cmd.CommandText = "INSERT INTO lexicon_spoken (word) VALUES ($w);";
                cmd.Parameters.AddWithValue("$w", word);
                cmd.ExecuteNonQuery();
            }

            foreach (var row in snap.QTable)
            {
                using var cmd = Connection.CreateCommand();
                cmd.Transaction = tx;
                cmd.CommandText = "INSERT INTO qtable (action, q, count) VALUES ($a, $q, $c);";
                cmd.Parameters.AddWithValue("$a", row[0]);
                cmd.Parameters.AddWithValue("$q", row[1]);
                cmd.Parameters.AddWithValue("$c", row[2]);
                cmd.ExecuteNonQuery();
            }

            foreach (var e in snap.Associations)
            {
                using var cmd = Connection.CreateCommand();
                cmd.Transaction = tx;
                cmd.CommandText = "INSERT OR REPLACE INTO associations (node_a, node_b, weight) VALUES ($a, $b, $w);";
                cmd.Parameters.AddWithValue("$a", e[0]);
                cmd.Parameters.AddWithValue("$b", e[1]);
                cmd.Parameters.AddWithValue("$w", e[2]);
                cmd.ExecuteNonQuery();
            }

            foreach (var s in snap.SemanticEntries)
            {
                using var cmd = Connection.CreateCommand();
                cmd.Transaction = tx;
                cmd.CommandText = "INSERT INTO semantic (text, vector, tag, created_at) VALUES ($t, $v, $g, $c);";
                cmd.Parameters.AddWithValue("$t", s.Text);
                cmd.Parameters.AddWithValue("$v", System.Text.Json.JsonSerializer.Serialize(s.Vector));
                cmd.Parameters.AddWithValue("$g", s.Tag ?? "");
                cmd.Parameters.AddWithValue("$c", s.CreatedAt.ToString("O"));
                cmd.ExecuteNonQuery();
            }

            foreach (var saying in snap.Sayings)
            {
                using var cmd = Connection.CreateCommand();
                cmd.Transaction = tx;
                cmd.CommandText = "INSERT INTO sayings (text) VALUES ($t);";
                cmd.Parameters.AddWithValue("$t", saying);
                cmd.ExecuteNonQuery();
            }

            if (!string.IsNullOrWhiteSpace(snap.LanguageModelJson))
            {
                using var cmd = Connection.CreateCommand();
                cmd.Transaction = tx;
                cmd.CommandText = "INSERT INTO langmodel (id, payload) VALUES (1, $p);";
                cmd.Parameters.AddWithValue("$p", snap.LanguageModelJson);
                cmd.ExecuteNonQuery();
            }

            if (!string.IsNullOrWhiteSpace(snap.NeuralLanguageModelJson))
            {
                using var cmd = Connection.CreateCommand();
                cmd.Transaction = tx;
                cmd.CommandText = "INSERT INTO deeplangmodel (id, payload) VALUES (1, $p);";
                cmd.Parameters.AddWithValue("$p", snap.NeuralLanguageModelJson);
                cmd.ExecuteNonQuery();
            }

            if (!string.IsNullOrWhiteSpace(snap.SelfImprovementJson))
            {
                using var cmd = Connection.CreateCommand();
                cmd.Transaction = tx;
                cmd.CommandText = "INSERT INTO selfimprovement (id, payload) VALUES (1, $p);";
                cmd.Parameters.AddWithValue("$p", snap.SelfImprovementJson);
                cmd.ExecuteNonQuery();
            }

            tx.Commit();
        }
        catch (SqliteException ex)
        {
            throw new PersistenceException($"Failed to save mind to '{Path}': {ex.Message}", ex);
        }
    }

    /// <summary>Loads a snapshot (synchronous). Throws if no mind is stored.</summary>
    public MindSnapshot Load()
    {
        try
        {
            if (!Exists())
                throw new PersistenceException($"No stored mind found in '{Path}'.");

            var snap = new MindSnapshot
            {
                Name = ReadValue("name") ?? "Restored",
                Description = ReadValue("description") ?? "",
                Cycles = int.TryParse(ReadValue("cycles"), out var cycles) ? cycles : 0,
                SavedAt = DateTimeOffset.TryParse(ReadValue("saved_at"), out var at) ? at : DateTimeOffset.UtcNow,
                ConfigJson = ReadValue("config") ?? "{}"
            };

            using (var cmd = Connection.CreateCommand())
            {
                cmd.CommandText = "SELECT emotion, intensity FROM emotions;";
                using var reader = cmd.ExecuteReader();
                while (reader.Read()) snap.Emotions[reader.GetString(0)] = reader.GetDouble(1);
            }

            using (var cmd = Connection.CreateCommand())
            {
                cmd.CommandText = "SELECT valence, arousal FROM mood LIMIT 1;";
                using var reader = cmd.ExecuteReader();
                if (reader.Read())
                {
                    snap.MoodValence = reader.GetDouble(0);
                    snap.MoodArousal = reader.GetDouble(1);
                }
            }

            using (var cmd = Connection.CreateCommand())
            {
                cmd.CommandText = @"SELECT kind, content, importance, strength, created_at, last_recalled_at,
                                           recall_count, emotion_type, emotion_intensity, source FROM memories;";
                using var reader = cmd.ExecuteReader();
                while (reader.Read())
                {
                    snap.Memories.Add(new MindSnapshot.MemoryDto
                    {
                        Kind = reader.GetString(0),
                        Content = reader.GetString(1),
                        Importance = reader.GetDouble(2),
                        Strength = reader.GetDouble(3),
                        CreatedAt = DateTimeOffset.Parse(reader.GetString(4)),
                        LastRecalledAt = reader.IsDBNull(5) ? null : DateTimeOffset.Parse(reader.GetString(5)),
                        RecallCount = reader.GetInt32(6),
                        EmotionType = reader.IsDBNull(7) ? null : reader.GetString(7),
                        EmotionIntensity = reader.GetDouble(8),
                        Source = reader.IsDBNull(9) ? null : reader.GetString(9)
                    });
                }
            }

            using (var cmd = Connection.CreateCommand())
            {
                cmd.CommandText = "SELECT subject, predicate, object, confidence, source FROM facts;";
                using var reader = cmd.ExecuteReader();
                while (reader.Read())
                    snap.Facts.Add(new[] { reader.GetString(0), reader.GetString(1), reader.GetString(2),
                        reader.GetDouble(3).ToString(System.Globalization.CultureInfo.InvariantCulture),
                        reader.IsDBNull(4) ? "" : reader.GetString(4) });
            }

            using (var cmd = Connection.CreateCommand())
            {
                cmd.CommandText = "SELECT payload FROM rules;";
                using var reader = cmd.ExecuteReader();
                while (reader.Read())
                {
                    var rule = System.Text.Json.JsonSerializer.Deserialize<MindSnapshot.RuleDto>(reader.GetString(0));
                    if (rule is not null) snap.Rules.Add(rule);
                }
            }

            using (var cmd = Connection.CreateCommand())
            {
                cmd.CommandText = "SELECT payload FROM goals;";
                using var reader = cmd.ExecuteReader();
                while (reader.Read())
                {
                    var goal = System.Text.Json.JsonSerializer.Deserialize<MindSnapshot.GoalDto>(reader.GetString(0));
                    if (goal is not null) snap.Goals.Add(goal);
                }
            }

            using (var cmd = Connection.CreateCommand())
            {
                cmd.CommandText = "SELECT name, level, growth FROM drives;";
                using var reader = cmd.ExecuteReader();
                while (reader.Read())
                    snap.Drives.Add(new[] { reader.GetString(0),
                        reader.GetDouble(1).ToString(System.Globalization.CultureInfo.InvariantCulture),
                        reader.GetDouble(2).ToString(System.Globalization.CultureInfo.InvariantCulture) });
            }

            using (var cmd = Connection.CreateCommand())
            {
                cmd.CommandText = "SELECT network, stats FROM learning LIMIT 1;";
                using var reader = cmd.ExecuteReader();
                if (reader.Read() && !reader.IsDBNull(0))
                {
                    snap.NetworkJson = reader.GetString(0);
                    snap.Stats = System.Text.Json.JsonSerializer.Deserialize<MindSnapshot.LearningStatsDto>(reader.GetString(1))
                                 ?? new MindSnapshot.LearningStatsDto();
                }
            }

            using (var cmd = Connection.CreateCommand())
            {
                cmd.CommandText = "SELECT action, q, count FROM qtable;";
                using var reader = cmd.ExecuteReader();
                while (reader.Read())
                    snap.QTable.Add(new[] { reader.GetInt32(0).ToString(),
                        reader.GetDouble(1).ToString(System.Globalization.CultureInfo.InvariantCulture),
                        reader.GetInt64(2).ToString() });
            }

            using (var cmd = Connection.CreateCommand())
            {
                cmd.CommandText = "SELECT node_a, node_b, weight FROM associations;";
                using var reader = cmd.ExecuteReader();
                while (reader.Read())
                    snap.Associations.Add(new[] { reader.GetString(0), reader.GetString(1),
                        reader.GetDouble(2).ToString(System.Globalization.CultureInfo.InvariantCulture) });
            }

            using (var cmd = Connection.CreateCommand())
            {
                cmd.CommandText = "SELECT text, vector, tag, created_at FROM semantic;";
                using var reader = cmd.ExecuteReader();
                while (reader.Read())
                {
                    try
                    {
                        snap.SemanticEntries.Add(new MindSnapshot.SemanticDto
                        {
                            Text = reader.GetString(0),
                            Vector = System.Text.Json.JsonSerializer.Deserialize<List<double>>(reader.GetString(1)) ?? new(),
                            Tag = reader.IsDBNull(2) ? "" : reader.GetString(2),
                            CreatedAt = DateTimeOffset.TryParse(reader.GetString(3), out var created) ? created : DateTimeOffset.UtcNow
                        });
                    }
                    catch (System.Text.Json.JsonException)
                    {
                        // Corrupt vector payload: skip the entry.
                    }
                }
            }

            if (TableExists("stylenet"))
            {
                using (var cmd = Connection.CreateCommand())
                {
                    cmd.CommandText = "SELECT network FROM stylenet LIMIT 1;";
                    var result = cmd.ExecuteScalar();
                    if (result is string styleNetJson) snap.StyleNetworkJson = styleNetJson;
                }
            }

            if (TableExists("lexicon"))
            {
                using (var cmd = Connection.CreateCommand())
                {
                    cmd.CommandText = "SELECT word, count, first_cycle, last_cycle FROM lexicon ORDER BY first_cycle;";
                    using var reader = cmd.ExecuteReader();
                    while (reader.Read())
                        snap.LexiconEntries.Add(new[] { reader.GetString(0),
                            reader.GetInt64(1).ToString(), reader.GetInt64(2).ToString(),
                            reader.GetInt64(3).ToString() });
                }
                using (var cmd = Connection.CreateCommand())
                {
                    cmd.CommandText = "SELECT word FROM lexicon_spoken;";
                    using var reader = cmd.ExecuteReader();
                    while (reader.Read())
                        snap.LexiconSpoken.Add(reader.GetString(0));
                }
            }

            if (TableExists("langmodel"))
            {
                using (var cmd = Connection.CreateCommand())
                {
                    cmd.CommandText = "SELECT payload FROM langmodel LIMIT 1;";
                    var result = cmd.ExecuteScalar();
                    if (result is string lmJson) snap.LanguageModelJson = lmJson;
                }
            }

            if (TableExists("deeplangmodel"))
            {
                using (var cmd = Connection.CreateCommand())
                {
                    cmd.CommandText = "SELECT payload FROM deeplangmodel LIMIT 1;";
                    var result = cmd.ExecuteScalar();
                    if (result is string nlmJson) snap.NeuralLanguageModelJson = nlmJson;
                }
            }

            if (TableExists("selfimprovement"))
            {
                using (var cmd = Connection.CreateCommand())
                {
                    cmd.CommandText = "SELECT payload FROM selfimprovement LIMIT 1;";
                    var result = cmd.ExecuteScalar();
                    if (result is string siJson) snap.SelfImprovementJson = siJson;
                }
            }

            using (var cmd = Connection.CreateCommand())
            {
                cmd.CommandText = "SELECT text FROM sayings ORDER BY id;";
                using var reader = cmd.ExecuteReader();
                while (reader.Read())
                    snap.Sayings.Add(reader.GetString(0));
            }

            return snap;
        }
        catch (SqliteException ex)
        {
            throw new PersistenceException($"Failed to load mind from '{Path}': {ex.Message}", ex);
        }
    }

    /// <summary>True when a mind profile is stored in this database.</summary>
    public bool Exists()
    {
        try
        {
            using var cmd = Connection.CreateCommand();
            cmd.CommandText = "SELECT count(*) FROM sqlite_master WHERE type='table' AND name='profile';";
            return Convert.ToInt64(cmd.ExecuteScalar()) > 0;
        }
        catch (SqliteException)
        {
            return false;
        }
    }

    /// <summary>True when the (possibly older) database has the given table.</summary>
    private bool TableExists(string table)
    {
        using var cmd = Connection.CreateCommand();
        cmd.CommandText = "SELECT count(*) FROM sqlite_master WHERE type='table' AND name=$t;";
        cmd.Parameters.AddWithValue("$t", table);
        return Convert.ToInt64(cmd.ExecuteScalar()) > 0;
    }

    private void Bind(string key, string value, SqliteTransaction tx)
    {
        using var cmd = Connection.CreateCommand();
        cmd.Transaction = tx;
        cmd.CommandText = "INSERT INTO profile (key, value) VALUES ($k, $v);";
        cmd.Parameters.AddWithValue("$k", key);
        cmd.Parameters.AddWithValue("$v", value);
        cmd.ExecuteNonQuery();
    }

    private string? ReadValue(string key)
    {
        using var cmd = Connection.CreateCommand();
        cmd.CommandText = "SELECT value FROM profile WHERE key = $k LIMIT 1;";
        cmd.Parameters.AddWithValue("$k", key);
        var result = cmd.ExecuteScalar();
        return result as string;
    }

    private void Execute(string sql, SqliteTransaction tx)
    {
        using var cmd = Connection.CreateCommand();
        cmd.Transaction = tx;
        cmd.CommandText = sql;
        cmd.ExecuteNonQuery();
    }

    /// <inheritdoc />
    public void Dispose()
    {
        _connection?.Dispose();
        _connection = null;
    }
}
