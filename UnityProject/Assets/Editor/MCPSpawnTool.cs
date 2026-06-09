using System;
using System.IO;
using Newtonsoft.Json.Linq;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.Networking;
using MCP.Editor; // CoplayDev/unity-mcp namespace

namespace ThreeDGenerator.Editor
{
    /// <summary>
    /// Custom MCP Tool registered with the CoplayDev/unity-mcp package.
    /// Discovered automatically via [McpForUnityTool] attribute (reflection on Editor assemblies).
    ///
    /// The Relay Agent calls:
    ///   tools/call { name: "spawn_glb_from_url", arguments: { url, name, scene } }
    ///
    /// Unity downloads the GLB from the URL, imports it via glTFast, and spawns it in the scene.
    /// </summary>
    [McpForUnityTool("spawn_glb_from_url", Group = "3d-generator")]
    public static class MCPSpawnTool
    {
        // ── MCP entry-point ────────────────────────────────────────────────────
        /// <summary>
        /// Required handler method discovered by the MCP package via reflection.
        /// Parameters: url (string), name (string, optional), scene (string: "new"|"existing")
        /// </summary>
        public static string HandleCommand(JObject parameters)
        {
            string url   = parameters["url"]?.Value<string>()   ?? "";
            string name  = parameters["name"]?.Value<string>()  ?? "";
            string scene = parameters["scene"]?.Value<string>() ?? "existing";

            if (string.IsNullOrEmpty(url))
                return "[MCPSpawnTool] ERROR: 'url' parameter is required.";

            // Queue the actual work on the main thread
            EditorApplication.delayCall += () => _DownloadAndSpawn(url, name, scene);

            return $"[MCPSpawnTool] Spawn queued: {url}";
        }

        // ── Internal logic ────────────────────────────────────────────────────
        static readonly string ImportDir      = Path.Combine(Application.dataPath, "Imported");
        const           string ImportDirAsset = "Assets/Imported";

        static void _DownloadAndSpawn(string url, string displayName, string scene)
        {
            Directory.CreateDirectory(ImportDir);

            string ext = "glb";
            try { ext = Path.GetExtension(new Uri(url).LocalPath).TrimStart('.').ToLower(); } catch { }
            if (string.IsNullOrEmpty(ext)) ext = "glb";

            string id        = Sanitize(string.IsNullOrEmpty(displayName) ? Guid.NewGuid().ToString() : displayName);
            string filePath  = Path.Combine(ImportDir, $"{id}.{ext}");
            string assetPath = $"{ImportDirAsset}/{id}.{ext}";

            Debug.Log($"[MCPSpawnTool] Downloading: {url}");
            var www = UnityWebRequest.Get(url);
            var op  = www.SendWebRequest();

            op.completed += _ =>
            {
                try
                {
                    if (www.result != UnityWebRequest.Result.Success)
                    {
                        Debug.LogError($"[MCPSpawnTool] Download failed: {www.error}");
                        return;
                    }

                    File.WriteAllBytes(filePath, www.downloadHandler.data);
                    AssetDatabase.ImportAsset(assetPath, ImportAssetOptions.ForceSynchronousImport);
                    AssetDatabase.Refresh();

                    var prefab = AssetDatabase.LoadAssetAtPath<GameObject>(assetPath);
                    if (prefab == null)
                    {
                        Debug.LogError($"[MCPSpawnTool] glTFast produced no prefab for {assetPath}. " +
                                       "Is com.unity.cloud.gltfast installed?");
                        return;
                    }

                    if (scene == "new")
                    {
                        var active = EditorSceneManager.GetActiveScene();
                        if (active.isDirty)
                            EditorSceneManager.SaveCurrentModifiedScenesIfUserWantsTo();
                        EditorSceneManager.NewScene(NewSceneSetup.DefaultGameObjects, NewSceneMode.Single);
                    }

                    var go = (GameObject)PrefabUtility.InstantiatePrefab(prefab);
                    if (go == null)
                    {
                        Debug.LogError($"[MCPSpawnTool] InstantiatePrefab returned null for {assetPath}");
                        return;
                    }

                    if (!string.IsNullOrEmpty(displayName)) go.name = displayName;
                    Selection.activeGameObject = go;

                    var sv = SceneView.lastActiveSceneView;
                    if (sv != null) sv.FrameSelected();
                    EditorWindow.FocusWindowIfItsOpen<SceneView>();

                    _EnsureLight();

                    Debug.Log($"[MCPSpawnTool] ✅ Spawn MCP: '{go.name}' (scene={scene})");
                }
                catch (Exception ex)
                {
                    Debug.LogException(ex);
                }
                finally
                {
                    www.Dispose();
                }
            };
        }

        static void _EnsureLight()
        {
            if (UnityEngine.Object.FindObjectOfType<Light>() != null) return;
            var lg    = new GameObject("Directional Light");
            var light = lg.AddComponent<Light>();
            light.type      = LightType.Directional;
            light.intensity = 1.0f;
            lg.transform.rotation = Quaternion.Euler(50f, -30f, 0f);
        }

        static string Sanitize(string s)
        {
            foreach (var c in Path.GetInvalidFileNameChars())
                s = s.Replace(c, '_');
            return s.Length > 60 ? s.Substring(0, 60) : s;
        }
    }
}
