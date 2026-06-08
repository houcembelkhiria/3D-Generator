using System;
using System.Collections.Generic;
using System.IO;
using UnityEditor;
using UnityEngine;

namespace ThreeDGenerator.Editor
{
    /// <summary>
    /// Editor window that tails the MCP activity log written by Backend/app/mcp_server.py.
    /// Open via menu: 3D Generator → MCP Monitor
    /// </summary>
    public class MCPMonitor : EditorWindow
    {
        // Path relative to UnityProject/ → ../../Backend/generated/mcp_activity.log
        static string LogPath => Path.GetFullPath(
            Path.Combine(Application.dataPath, "..", "..", "Backend", "generated", "mcp_activity.log"));

        readonly List<LogEntry> _entries = new();
        Vector2 _scroll;
        double _nextRefresh;
        long _lastFilePos;
        string _filter = "";

        static readonly Color ColSpawn   = new(0.4f, 1.0f, 0.6f);
        static readonly Color ColGenerate = new(0.5f, 0.8f, 1.0f);
        static readonly Color ColStatus  = new(1.0f, 1.0f, 0.5f);
        static readonly Color ColDefault = new(0.85f, 0.85f, 0.85f);

        [Serializable]
        class LogEntry
        {
            public string ts;
            public string tool;
            public string raw;
        }

        [MenuItem("3D Generator/MCP Monitor")]
        static void Open()
        {
            var w = GetWindow<MCPMonitor>("MCP Monitor");
            w.minSize = new Vector2(420, 260);
            w.Show();
        }

        void OnEnable()
        {
            _entries.Clear();
            _lastFilePos = 0;
            RefreshLog();
            EditorApplication.update += OnEditorUpdate;
        }

        void OnDisable() => EditorApplication.update -= OnEditorUpdate;

        void OnEditorUpdate()
        {
            if (EditorApplication.timeSinceStartup < _nextRefresh) return;
            _nextRefresh = EditorApplication.timeSinceStartup + 1.0;
            RefreshLog();
        }

        void RefreshLog()
        {
            var fi = new FileInfo(LogPath);
            if (!fi.Exists || fi.Length == _lastFilePos) return;

            using var fs = new FileStream(LogPath, FileMode.Open, FileAccess.Read, FileShare.ReadWrite);
            fs.Seek(_lastFilePos, SeekOrigin.Begin);
            using var sr = new StreamReader(fs);
            string line;
            bool added = false;
            while ((line = sr.ReadLine()) != null)
            {
                if (string.IsNullOrWhiteSpace(line)) continue;
                var entry = ParseEntry(line);
                _entries.Add(entry);
                added = true;
            }
            _lastFilePos = fs.Position;
            if (added) Repaint();
        }

        static LogEntry ParseEntry(string raw)
        {
            var e = new LogEntry { raw = raw, ts = "?", tool = "?" };
            try
            {
                // Minimal JSON field extraction without a full JSON parser dependency
                e.ts   = ExtractJsonString(raw, "ts")   ?? "?";
                e.tool = ExtractJsonString(raw, "tool") ?? "?";
            }
            catch { /* keep defaults */ }
            return e;
        }

        static string ExtractJsonString(string json, string key)
        {
            var search = $"\"{key}\":\"";
            int start = json.IndexOf(search, StringComparison.Ordinal);
            if (start < 0) return null;
            start += search.Length;
            int end = json.IndexOf('"', start);
            return end < 0 ? null : json.Substring(start, end - start);
        }

        void OnGUI()
        {
            DrawToolbar();

            var filtered = string.IsNullOrEmpty(_filter)
                ? _entries
                : _entries.FindAll(e => e.raw.Contains(_filter, StringComparison.OrdinalIgnoreCase));

            _scroll = EditorGUILayout.BeginScrollView(_scroll, GUILayout.ExpandHeight(true));
            var style = new GUIStyle(EditorStyles.label)
                { wordWrap = true, fontSize = 11, richText = false };

            for (int i = filtered.Count - 1; i >= 0; i--)
            {
                var e = filtered[i];
                GUI.color = RowColor(e.tool);
                EditorGUILayout.LabelField($"[{e.ts}] {e.tool}  {Summarize(e.raw)}", style);
                GUI.color = Color.white;
            }
            EditorGUILayout.EndScrollView();

            EditorGUILayout.LabelField(LogPath, EditorStyles.miniLabel);
            EditorGUILayout.LabelField(
                $"{_entries.Count} total entries • refreshes every 1 s",
                EditorStyles.miniLabel);
        }

        void DrawToolbar()
        {
            EditorGUILayout.BeginHorizontal(EditorStyles.toolbar);
            GUILayout.Label("MCP Activity", EditorStyles.boldLabel, GUILayout.Width(90));
            GUILayout.Label("Filter:", GUILayout.Width(38));
            _filter = EditorGUILayout.TextField(_filter, EditorStyles.toolbarTextField,
                GUILayout.ExpandWidth(true));
            if (GUILayout.Button("Clear", EditorStyles.toolbarButton, GUILayout.Width(44)))
            {
                _entries.Clear();
                _lastFilePos = 0;
            }
            if (GUILayout.Button("↺", EditorStyles.toolbarButton, GUILayout.Width(24)))
            {
                _entries.Clear();
                _lastFilePos = 0;
                RefreshLog();
            }
            EditorGUILayout.EndHorizontal();
        }

        static Color RowColor(string tool) => tool switch
        {
            "spawn_in_unity"    => ColSpawn,
            "generate_and_spawn" => ColSpawn,
            "generate_3d_from_text"      => ColGenerate,
            "generate_3d_from_image_url" => ColGenerate,
            "get_generation_status"      => ColStatus,
            _ => ColDefault,
        };

        static string Summarize(string raw)
        {
            // Show the args + result fields compactly
            int argsIdx = raw.IndexOf("\"args\":", StringComparison.Ordinal);
            if (argsIdx < 0) return "";
            return raw.Substring(argsIdx).Replace("\n", " ");
        }
    }
}
