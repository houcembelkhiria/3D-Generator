using System;

namespace ThreeDGenerator.Editor
{
    [Serializable]
    public class SpawnRequest
    {
        public string id;
        public string url;
        public string scene;
        public string name;
        public bool hasTexture;
    }
}
