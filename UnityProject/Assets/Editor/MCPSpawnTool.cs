using System;
using System.Collections;
using System.IO;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.Networking;

namespace ThreeDGenerator.Editor
{
    /// <summary>
    /// Custom MCP Tool exposed to the unity-mcp server.
    /// Registered automatically via [MCPTool] when the CoplayDev/unity-mcp package
    /// is installed. The Relay Agent calls:
    ///   tools/call { name: "spawn_glb_from_url", arguments: { url, name, scene } }
    /// Unity downloads the GLB, imports it via glTFast, and spawns it in the scene.
    /// </summary>
    public static class MCPSpawnTool
    {
        // ── State ─────────────────────────────────────────────────────────────
        static readonly string ImportDir     = Path.Combine(Application.dataPath, "Imported");
        const           string ImportDirAsset = "Assets/Imported";

        // ── MCP Tool entry-point ──────────────────────────────────────────────
        /// <summary>
        /// MCP tool: download a GLB/GLTF from <paramref name="url"/>,
        /// import it into the project, and spawn it in the scene.
        /// </summary>
        /// <param name="url">Public HTTP(S) URL of the .glb file.</param>
        /// <param name="name">Display name for the spawned GameObject.</param>
        /// <param name="scene">"new" to open a blank scene first, "existing" to keep current.</param>
        /// <returns>Result message logged back to the MCP caller.</returns>
        // NOTE: The [MCPTool] attribute is injected by the com.coplay.mcp-for-unity package.
        // If the package uses a different attribute name, rename accordingly after install.
        [UnityEngine.RuntimeInitializeOnLoadMethod]   // keeps compiler happy until package resolves
        public static void _RegisterTool() { /* no-op — package reflection picks up SpawnGLBFromUrl */ }

        /// <summary>Actual implementation — called by the MCP dispatcher via reflection.</summary>
        public static string SpawnGLBFromUrl(string url, string name = "", string scene = "existing")
        {
            try
            {
                Directory.CreateDirectory(ImportDir);

                // Derive a safe asset ID from url
                string ext  = "glb";
                try { ext = Path.GetExtension(new Uri(url).LocalPath).TrimStart('.').ToLower(); } catch { }
                if (string.IsNullOrEmpty(ext)) ext = "glb";

                string id        = Sanitize(string.IsNullOrEmpty(name) ? Guid.NewGuid().ToString() : name);
                string filePath  = Path.Combine(ImportDir, $"{id}.{ext}");
                string assetPath = $"{ImportDirAsset}/{id}.{ext}";

                // Kick off coroutine-equivalent via EditorApplication.delayCall chain
                EditorApplication.delayCall += () => _DownloadAndSpawn(url, filePath, assetPath, name, scene);

                return $"[MCPSpawnTool] Spawn queued: downloading {url}";
            }
            catch (Exception ex)
            {
                Debug.LogError($"[MCPSpawnTool] SpawnGLBFromUrl failed: {ex}");
                return $"[MCPSpawnTool] ERROR: {ex.Message}";
            }
        }

        // ── Internal helpers ──────────────────────────────────────────────────
        static void _DownloadAndSpawn(string url, string filePath, string assetPath,
                                      string displayName, string scene)
        {
            Debug.Log($"[MCPSpawnTool] Downloading {url}");
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
                                       "Ensure com.unity.cloud.gltfast is installed.");
                        return;
                    }

                    // Optionally open a new scene
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

                    // Ensure a light exists for textured models
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
