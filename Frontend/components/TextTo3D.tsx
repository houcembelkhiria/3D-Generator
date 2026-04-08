import React, { useState } from 'react';
import { GeneratedModel } from '../types';
import { ModelViewer3D } from './ModelViewer3D';
import { IconMessageSquare, IconLoader, IconBox } from './Icons';
import { API_BASE } from '../api';


interface TextTo3DProps {
  onModelGenerated?: (model: GeneratedModel) => void;
}

export const TextTo3D: React.FC<TextTo3DProps> = ({ onModelGenerated }) => {
  const [prompt, setPrompt] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<{ previewUrl: string; downloadUrl: string } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [texture, setTexture] = useState(false);
  const [steps, setSteps] = useState(5);
  const [outputType, setOutputType] = useState('glb');

  const generate = async () => {
    if (!prompt.trim()) return;
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const res = await fetch(`${API_BASE}/api/v1/text-to-3d`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text: prompt,
          num_inference_steps: steps,
          guidance_scale: 5.0,
          octree_resolution: 128,
          num_chunks: 8000,
          texture,
          type: outputType,
        }),
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || `HTTP ${res.status}`);
      }

      const data = await res.json();
      setResult({
        previewUrl: `${API_BASE}${data.preview_url}`,
        downloadUrl: `${API_BASE}${data.download_url}`,
      });

      onModelGenerated?.({
        id: crypto.randomUUID(),
        previewUrl: `${API_BASE}${data.preview_url}`,
        downloadUrl: `${API_BASE}${data.download_url}`,
        format: data.format,
        source: 'text-to-3d',
        prompt,
        createdAt: new Date().toISOString(),
        fromCache: data.from_cache ?? false,
        attempt: data.attempt,
      });
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-5xl mx-auto space-y-6">
      <div className="card p-6">
        <h2 className="text-xl font-bold text-heading mb-2 flex items-center">
          <IconMessageSquare className="mr-3 text-[#FF8C66]" /> Text to 3D
        </h2>
        <p className="text-body text-sm mb-6">Describe an object and generate a 3D model from text.</p>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Left: Input */}
          <div className="space-y-4">
            <textarea
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              placeholder="Describe the 3D object... (e.g. 'a red sports car', 'a medieval sword')"
              rows={6}
              className="w-full bg-[var(--bg-input)] border border-theme-secondary rounded-xl p-4 text-sm text-theme-primary placeholder-[var(--text-muted)] focus:outline-none focus:ring-1 focus:ring-[#FF8C66] focus:border-[#FF8C66] resize-none"
              disabled={loading}
            />

            {/* Options */}
            <div className="flex items-center gap-4">
              <label className="flex items-center gap-2 text-sm text-theme-secondary">
                <input type="checkbox" checked={texture} onChange={(e) => setTexture(e.target.checked)} className="accent-[#7C3AED]" />
                Generate texture
              </label>

              <label className="flex items-center gap-2 text-sm text-theme-secondary">
                Format:
                <select value={outputType} onChange={(e) => setOutputType(e.target.value)} className="bg-[var(--bg-input)] border border-theme rounded px-2 py-1 text-sm text-theme-primary">
                  <option value="glb">GLB</option>
                  <option value="obj">OBJ</option>
                  <option value="ply">PLY</option>
                  <option value="stl">STL</option>
                </select>
              </label>
              <label className="flex items-center gap-2 text-sm text-theme-secondary">
                Steps:
                <input type="number" value={steps} onChange={(e) => setSteps(Number(e.target.value))} min={1} max={100} className="w-16 bg-[var(--bg-input)] border border-theme rounded px-2 py-1 text-sm text-theme-primary" />
              </label>
            </div>

            <button
              onClick={generate}
              disabled={!prompt.trim() || loading}
              className="w-full px-4 py-3 bg-[#FF8C66] hover:bg-[#ff7a4d] disabled:opacity-50 disabled:cursor-not-allowed text-black rounded-xl font-bold transition-all flex items-center justify-center gap-2"
            >
              {loading ? <><IconLoader className="w-5 h-5 animate-spin" /> Generating...</> : 'Generate 3D Model'}
            </button>

            {error && <p className="text-red-400 text-sm bg-red-500/10 border border-red-500/20 p-3 rounded-lg">{error}</p>}

            {loading && (
              <div className="text-theme-muted text-xs bg-[var(--bg-tertiary)] p-3 rounded-lg border border-theme">
                Text-to-3D first generates an image via HunyuanDiT, then converts it to a 3D mesh. This can take 1-3 minutes.
              </div>
            )}
          </div>

          {/* Right: Result */}
          <div className="space-y-4">
            {result ? (
              <>
                <ModelViewer3D src={result.previewUrl} />
                <a
                  href={result.downloadUrl}
                  download
                  className="block w-full text-center px-4 py-3 bg-[#FF8C66] hover:bg-[#ff7a4d] text-black rounded-xl font-bold transition-all"
                >
                  {`Download ${outputType.toUpperCase()}`}
                </a>
              </>
            ) : (
              <div className="flex flex-col items-center justify-center min-h-[300px] bg-[var(--bg-tertiary)] rounded-xl border border-theme text-theme-muted">
                <IconBox className="w-16 h-16 mb-4 opacity-20" />
                <p>3D preview will appear here</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
