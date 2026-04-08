import React, { useState } from 'react';
import { GeneratedModel } from '../types';
import { ModelViewer3D } from './ModelViewer3D';
import { IconBox, IconTrash, IconX } from './Icons';

interface ModelGalleryProps {
  models: GeneratedModel[];
  onRemove?: (id: string) => void;
}

const sourceLabels: Record<GeneratedModel['source'], string> = {
  'image-to-3d': 'Image',
  'text-to-3d': 'Text',
  'multiview-to-3d': 'Multi-View',
};

const sourceColors: Record<GeneratedModel['source'], string> = {
  'image-to-3d': 'bg-[#FF8C66]/20 text-[#FF8C66]',
  'text-to-3d': 'bg-[#7C3AED]/20 text-[#7C3AED]',
  'multiview-to-3d': 'bg-emerald-500/20 text-emerald-400',
};

export const ModelGallery: React.FC<ModelGalleryProps> = ({ models, onRemove }) => {
  const [selectedModel, setSelectedModel] = useState<GeneratedModel | null>(null);

  if (models.length === 0) {
    return (
      <div className="max-w-5xl mx-auto">
        <div className="card p-6">
          <h2 className="text-xl font-bold text-heading mb-6 flex items-center">
            <IconBox className="mr-3 text-[#FF8C66]" /> Model Gallery
          </h2>
          <div className="flex flex-col items-center justify-center py-16 text-theme-muted">
            <IconBox className="w-20 h-20 mb-4 opacity-20" />
            <p className="text-lg font-medium">No models generated yet</p>
            <p className="text-sm mt-1">Use Image-to-3D, Text-to-3D, or Multi-View to generate models</p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-6xl mx-auto space-y-6">
      <div className="card p-6">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-xl font-bold text-heading flex items-center">
            <IconBox className="mr-3 text-[#FF8C66]" /> Model Gallery
          </h2>
          <span className="text-sm text-theme-muted">{models.length} model{models.length !== 1 ? 's' : ''}</span>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {models.map(model => (
            <div
              key={model.id}
              className="bg-[var(--bg-tertiary)] rounded-xl border border-theme hover:border-[var(--border-secondary)] transition-all cursor-pointer group overflow-hidden"
              onClick={() => setSelectedModel(model)}
            >
              {/* Thumbnail via model-viewer */}
              <div className="aspect-square bg-[var(--bg-input)] relative overflow-hidden">
                {/* @ts-ignore */}
                <model-viewer
                  src={model.previewUrl}
                  auto-rotate
                  camera-controls
                  style={{ width: '100%', height: '100%', backgroundColor: 'transparent' }}
                  interaction-prompt="none"
                />
              </div>

              <div className="p-3 space-y-2">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-1.5">
                    <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${sourceColors[model.source]}`}>
                      {sourceLabels[model.source]}
                    </span>
                    {model.fromCache && (
                      <span className="text-xs px-2 py-0.5 rounded-full font-medium bg-amber-500/20 text-amber-400">
                        Cached
                      </span>
                    )}
                    {model.attempt && model.attempt > 1 && (
                      <span className="text-xs px-2 py-0.5 rounded-full font-medium bg-emerald-500/20 text-emerald-400">
                        v{model.attempt}
                      </span>
                    )}
                  </div>
                  <span className="text-xs text-theme-muted uppercase">{model.format}</span>
                </div>

                {model.prompt && (
                  <p className="text-xs text-theme-secondary truncate">{model.prompt}</p>
                )}

                <div className="flex items-center justify-between">
                  <span className="text-xs text-theme-muted">
                    {new Date(model.createdAt).toLocaleTimeString()}
                  </span>
                  <div className="flex gap-1">
                    <a
                      href={model.downloadUrl}
                      download
                      onClick={(e) => e.stopPropagation()}
                      className="text-xs px-2 py-1 bg-[#7C3AED] hover:bg-[#6d28d9] text-white rounded transition-colors"
                    >
                      Download
                    </a>
                    {onRemove && (
                      <button
                        onClick={(e) => { e.stopPropagation(); onRemove(model.id); }}
                        className="p-1 text-theme-muted hover:text-red-400 transition-colors"
                      >
                        <IconTrash className="w-3.5 h-3.5" />
                      </button>
                    )}
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Full-screen viewer modal */}
      {selectedModel && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm" onClick={() => setSelectedModel(null)}>
          <div className="bg-[var(--bg-card)] rounded-2xl border border-theme shadow-2xl w-full max-w-3xl m-4 overflow-hidden" onClick={(e) => e.stopPropagation()}>
            <div className="flex items-center justify-between p-4 border-b border-theme">
              <div className="flex items-center gap-3">
                <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${sourceColors[selectedModel.source]}`}>
                  {sourceLabels[selectedModel.source]}
                </span>
                {selectedModel.prompt && <span className="text-sm text-theme-secondary">{selectedModel.prompt}</span>}
              </div>
              <button onClick={() => setSelectedModel(null)} className="p-2 text-theme-muted hover:text-theme-primary transition-colors">
                <IconX className="w-5 h-5" />
              </button>
            </div>
            <div className="p-4">
              <ModelViewer3D src={selectedModel.previewUrl} className="!aspect-[4/3]" />
            </div>
            <div className="flex gap-3 p-4 border-t border-theme">
              <a
                href={selectedModel.downloadUrl}
                download
                className="flex-1 text-center px-4 py-2 bg-[#7C3AED] hover:bg-[#6d28d9] text-white rounded-xl font-bold transition-all"
              >
                Download {selectedModel.format.toUpperCase()}
              </a>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
