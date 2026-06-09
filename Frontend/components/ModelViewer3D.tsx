import React, { useState } from 'react';
import { IconBox } from './Icons';

interface ModelViewer3DProps {
  src: string;
  alt?: string;
  className?: string;
  cameraOrbit?: string;
}

export const ModelViewer3D: React.FC<ModelViewer3DProps> = ({ src, alt = '3D Model', className = '', cameraOrbit }) => {
  const [error, setError] = useState(false);

  if (error) {
    return (
      <div className={`relative w-full aspect-square bg-[var(--bg-tertiary)] rounded-xl border border-theme overflow-hidden flex flex-col items-center justify-center text-theme-muted ${className}`}>
        <IconBox className="w-12 h-12 mb-3 opacity-40" />
        <p className="text-sm">3D preview unavailable</p>
        <a href={src} download className="mt-2 text-xs text-[#FF8C66] underline">Download file directly</a>
      </div>
    );
  }

  return (
    <div className={`relative w-full aspect-square bg-[var(--bg-tertiary)] rounded-xl border border-theme overflow-hidden ${className}`}>
      {/* @ts-ignore — model-viewer is a web component loaded via CDN */}
      {/* key=src forces a full DOM remount on every new model, preventing the
          previous mesh from leaking into the next generation's viewer */}
      <model-viewer
        key={src}
        src={src}
        alt={alt}
        auto-rotate
        camera-controls
        shadow-intensity="1"
        ar-status="not-presenting"
        loading="eager"
        camera-orbit={cameraOrbit ?? '0deg 75deg 105%'}
        onError={() => setError(true)}
        style={{ width: '100%', height: '100%', backgroundColor: 'transparent' }}
      />
    </div>
  );
};
