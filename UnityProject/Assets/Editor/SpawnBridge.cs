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
                            EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);
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
                        Debug.Log($"[SpawnBridge] spawned '{go.name}' (scene={req.scene ?? "existing"})");
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
