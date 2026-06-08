using System;
using System.Collections.Generic;
using System.IO;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.Networking;

namespace ThreeDGenerator.Editor
{
    [InitializeOnLoad]
    public static class SpawnBridge
    {
        static readonly string ReqDir = Path.Combine(Application.dataPath, "SpawnRequests");
        static readonly string ImpDir = Path.Combine(Application.dataPath, "Imported");
        const string ImpAssetDir = "Assets/Imported";

        static readonly HashSet<string> InFlight = new HashSet<string>();
        static double _nextPoll;

        static SpawnBridge()
        {
            Directory.CreateDirectory(ReqDir);
            Directory.CreateDirectory(ImpDir);
            // Purge requests that pre-date this Editor session (e.g. left over
            // from a crash). Anything older than 60 s at startup is stale.
            try
            {
                var cutoff = DateTime.UtcNow.AddSeconds(-60);
                foreach (var f in Directory.GetFiles(ReqDir, "*.json"))
                    if (File.GetLastWriteTimeUtc(f) < cutoff)
                        File.Delete(f);
            }
            catch { }
            EditorApplication.update += Tick;
        }

        static void Tick()
        {
            if (EditorApplication.timeSinceStartup < _nextPoll) return;
            _nextPoll = EditorApplication.timeSinceStartup + 0.5;

            string[] files;
            try { files = Directory.GetFiles(ReqDir, "*.json"); }
            catch { return; }

            foreach (var f in files)
            {
                if (InFlight.Contains(f)) continue;
                InFlight.Add(f);
                EditorApplication.delayCall += () => Process(f);
            }
        }

        static void Process(string reqPath)
        {
            try
            {
                var json = File.ReadAllText(reqPath);
                var req = JsonUtility.FromJson<SpawnRequest>(json);
                if (req == null || string.IsNullOrEmpty(req.url))
                {
                    Debug.LogError($"[SpawnBridge] invalid request: {reqPath}");
                    SafeDelete(reqPath);
                    return;
                }

                var id = string.IsNullOrEmpty(req.id)
                    ? DateTime.UtcNow.Ticks.ToString()
                    : Sanitize(req.id);
                // Derive the file extension from the URL so we correctly handle
                // whatever format the backend served (glb, gltf, obj, …).
                // glTFast handles glb/gltf; other formats fall back to Unity's
                // built-in importers. The URL is always an absolute http URL.
                string urlExt = "glb";
                try { urlExt = System.IO.Path.GetExtension(new Uri(req.url).LocalPath).TrimStart('.').ToLower(); } catch { }
                if (string.IsNullOrEmpty(urlExt)) urlExt = "glb";
                var glbPath = Path.Combine(ImpDir, $"{id}.{urlExt}");
                var assetPath = $"{ImpAssetDir}/{id}.{urlExt}";

                Debug.Log($"[SpawnBridge] downloading {req.url}");
                var www = UnityWebRequest.Get(req.url);
                var op = www.SendWebRequest();
                op.completed += _ =>
                {
                    try
                    {
                        if (www.result != UnityWebRequest.Result.Success)
                        {
                            Debug.LogError($"[SpawnBridge] download failed: {www.error}");
                            return;
                        }
                        File.WriteAllBytes(glbPath, www.downloadHandler.data);
                        AssetDatabase.ImportAsset(assetPath, ImportAssetOptions.ForceSynchronousImport);
                        // Flush glTFast sub-assets (textures, materials) before loading prefab
                        if (req.hasTexture)
                            AssetDatabase.Refresh();

                        var prefab = AssetDatabase.LoadAssetAtPath<GameObject>(assetPath);
                        if (prefab == null)
                        {
                            Debug.LogError($"[SpawnBridge] glTFast did not produce a prefab for {assetPath} — is com.unity.cloud.gltfast installed?");
                            return;
                        }

                        if (req.scene == "new")
                        {
                            var active = EditorSceneManager.GetActiveScene();
                            if (active.isDirty)
                                EditorSceneManager.SaveCurrentModifiedScenesIfUserWantsTo();
                            EditorSceneManager.NewScene(NewSceneSetup.DefaultGameObjects, NewSceneMode.Single);
                        }

                        var go = (GameObject)PrefabUtility.InstantiatePrefab(prefab);
                        if (go == null)
                        {
                            Debug.LogError($"[SpawnBridge] PrefabUtility.InstantiatePrefab returned null for {assetPath}");
                            return;
                        }
                        if (!string.IsNullOrEmpty(req.name)) go.name = req.name;
                        Selection.activeGameObject = go;
                        var sv = SceneView.lastActiveSceneView;
                        if (sv != null) sv.FrameSelected();
                        EditorWindow.FocusWindowIfItsOpen<SceneView>();
                        if (req.hasTexture)
                            _EnsureLight();
                        Debug.Log($"[SpawnBridge] spawned '{go.name}' (scene={req.scene ?? "existing"}, hasTexture={req.hasTexture})");
                if (req.hasTexture)
                    _ValidateTextures(go);
                    }
                    catch (Exception ex)
                    {
                        Debug.LogException(ex);
                    }
                    finally
                    {
                        www.Dispose();
                        SafeDelete(reqPath);
                        InFlight.Remove(reqPath);
                    }
                };
            }
            catch (Exception ex)
            {
                Debug.LogException(ex);
                SafeDelete(reqPath);
                InFlight.Remove(reqPath);
            }
        }

        static void _EnsureLight()
        {
            if (UnityEngine.Object.FindObjectOfType<Light>() != null) return;
            var lg = new GameObject("Directional Light");
            var light = lg.AddComponent<Light>();
            light.type = LightType.Directional;
            light.intensity = 1.0f;
            lg.transform.rotation = Quaternion.Euler(50f, -30f, 0f);
            Debug.Log("[SpawnBridge] added Directional Light so textured model renders correctly");
        }

        static void _ValidateTextures(GameObject go)
        {
            foreach (var rend in go.GetComponentsInChildren<Renderer>(true))
            {
                foreach (var mat in rend.sharedMaterials)
                {
                    if (mat != null && mat.mainTexture != null)
                        return; // at least one texture found — all good
                }
            }
            Debug.LogWarning($"[SpawnBridge] '{go.name}' was generated with texture but no texture is visible on its materials. " +
                             "The GLB may have been generated without texture, or glTFast could not extract the embedded textures.");
        }

        static void SafeDelete(string path)
        {
            try { if (File.Exists(path)) File.Delete(path); } catch { }
        }

        static string Sanitize(string s)
        {
            foreach (var c in Path.GetInvalidFileNameChars())
                s = s.Replace(c, '_');
            return s;
        }
    }
}
