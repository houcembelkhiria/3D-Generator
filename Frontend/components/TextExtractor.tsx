import React, { useState, useRef, ChangeEvent, useEffect } from 'react';
import { GeneratedModel } from '../types';
import { API_BASE } from '../api';

interface TextExtractionResult {
  filename: string;
  extracted_text: string;
  message?: string;
}

interface TextExtractorProps {
  onExtractComplete?: (result: TextExtractionResult) => void;
  onModelGenerated?: (model: GeneratedModel) => void;
}

interface ToastNotification {
  id: string;
  message: string;
  type: 'success' | 'error' | 'info' | 'warning';
}

export const TextExtractor: React.FC<TextExtractorProps> = ({ onExtractComplete, onModelGenerated }) => {
  const [file, setFile] = useState<File | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<TextExtractionResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [dragActive, setDragActive] = useState(false);
  const [progress, setProgress] = useState<number>(0);
  const [toasts, setToasts] = useState<ToastNotification[]>([]);
  // 3D generation from document
  const [generating3D, setGenerating3D] = useState(false);
  const [gen3DError, setGen3DError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const addToast = (message: string, type: ToastNotification['type'] = 'info') => {
    const toast: ToastNotification = {
      id: Math.random().toString(36).substring(7),
      message,
      type
    };
    setToasts(prev => [...prev, toast]);
    setTimeout(() => {
      removeToast(toast.id);
    }, 3000);
  };

  const removeToast = (id: string) => {
    setToasts(prev => prev.filter(toast => toast.id !== id));
  };

  const getFileTypeDescription = (file: File): string => {
    const typeMap: Record<string, string> = {
      'application/pdf': 'PDF Document',
      'message/rfc822': 'Email Message (.eml)',
      'text/plain': 'Plain Text File'
    };
    return typeMap[file.type] || 'Unknown File Type';
  };

  useEffect(() => {
    if (result && onExtractComplete) {
      onExtractComplete(result);
    }
  }, [result, onExtractComplete]);

  const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    const selectedFile = event.target.files?.[0];
    if (selectedFile) {
      validateAndSetFile(selectedFile);
    }
  };

  const validateAndSetFile = (file: File) => {
    const validTypes = ['application/pdf', 'message/rfc822'];
    if (!validTypes.includes(file.type)) {
      setError('Please select a PDF or EML file.');
      addToast('Unsupported file type. Please select a PDF or EML file.', 'error');
      return;
    }
    const maxSize = 50 * 1024 * 1024;
    if (file.size > maxSize) {
      const errorMessage = `File is too large. Maximum size is 50MB. Your file is ${(file.size / (1024 * 1024)).toFixed(1)}MB.`;
      setError(errorMessage);
      addToast(errorMessage, 'error');
      return;
    }
    const fileTypeDesc = getFileTypeDescription(file);
    addToast(`📁 ${fileTypeDesc} detected: ${file.name}`, 'success');
    setFile(file);
    setError(null);
    setResult(null);
    setProgress(0);
    setGen3DError(null);
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      validateAndSetFile(e.dataTransfer.files[0]);
    }
  };

  const handleDrag = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleExtractText = async () => {
    if (!file) {
      setError('Please select a PDF or EML file first.');
      return;
    }
    setIsLoading(true);
    setError(null);
    setResult(null);
    setProgress(0);
    try {
      const formData = new FormData();
      formData.append('file', file);
      const progressInterval = setInterval(() => {
        setProgress(prev => Math.min(prev + 10, 90));
      }, 200);
      const response = await fetch(`${API_BASE}/api/v1/extract-text/`, {
        method: 'POST',
        body: formData,
      });
      clearInterval(progressInterval);
      setProgress(100);
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to extract text from document');
      }
      const data: TextExtractionResult = await response.json();
      setResult(data);
    } catch (err) {
      console.error('Text extraction error:', err);
      setError(err instanceof Error ? err.message : 'An unknown error occurred');
    } finally {
      setIsLoading(false);
      setProgress(0);
    }
  };

  const handleGenerate3D = async () => {
    if (!file) return;
    setGenerating3D(true);
    setGen3DError(null);
    addToast('Starting 3D generation from document…', 'info');
    try {
      const formData = new FormData();
      formData.append('file', file);
      const res = await fetch(`${API_BASE}/api/v1/run-pipeline`, {
        method: 'POST',
        body: formData,
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || `HTTP ${res.status}`);
      }
      const { task_id } = await res.json();
      addToast('Pipeline started, polling for result…', 'info');

      // Poll until done
      let data: any = null;
      for (let attempt = 0; attempt < 120; attempt++) {
        await new Promise(r => setTimeout(r, 3000));
        const pollRes = await fetch(`${API_BASE}/api/v1/task/${task_id}`);
        if (!pollRes.ok) throw new Error(`Poll failed: HTTP ${pollRes.status}`);
        const poll = await pollRes.json();
        if (poll.status === 'completed') { data = poll; break; }
        if (poll.status === 'failed') throw new Error(poll.result?.error || 'Pipeline failed');
      }
      if (!data) throw new Error('Pipeline timed out');

      const modelResult = data.result;
      const model: GeneratedModel = {
        id: task_id,
        previewUrl: modelResult?.preview_url?.startsWith('http')
          ? modelResult.preview_url
          : `${API_BASE}${modelResult?.preview_url ?? ''}`,
        downloadUrl: modelResult?.download_url?.startsWith('http')
          ? modelResult.download_url
          : `${API_BASE}${modelResult?.download_url ?? ''}`,
        format: modelResult?.format ?? 'glb',
        source: 'image-to-3d',
        prompt: file.name,
        createdAt: new Date().toISOString(),
        fromCache: modelResult?.from_cache ?? false,
        generationTime: modelResult?.generation_time,
        faceCount: modelResult?.face_count,
        fileSizeMb: modelResult?.file_size_mb,
      };
      onModelGenerated?.(model);
      addToast('3D model generated and added to gallery!', 'success');
    } catch (e: any) {
      setGen3DError(e.message);
      addToast(`3D generation failed: ${e.message}`, 'error');
    } finally {
      setGenerating3D(false);
    }
  };

  const triggerFileSelect = () => {
    fileInputRef.current?.click();
  };

  const handleReset = () => {
    setFile(null);
    setResult(null);
    setError(null);
    setProgress(0);
    setGen3DError(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handleClearResult = () => {
    setResult(null);
    setFile(null);
    setProgress(0);
    setGen3DError(null);
  };

  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const getTextStats = (text: string) => {
    const lines = text.split('\n').filter(line => line.trim()).length;
    const words = text.trim() ? text.trim().split(/\s+/).length : 0;
    return { lines, words };
  };

  return (
    <div className="bg-[var(--bg-card)] p-6 rounded-2xl border border-theme">
      <h2 className="text-lg font-bold text-heading mb-4 flex items-center">
        <svg xmlns="http://www.w3.org/2000/svg" className="w-5 h-5 mr-2 text-[#7C3AED]" viewBox="0 0 20 20" fill="currentColor">
          <path fillRule="evenodd" d="M4 4a2 2 0 012-2h4.586A2 2 0 0112 2.586L15.414 6A2 2 0 0116 7.414V16a2 2 0 01-2 2H6a2 2 0 01-2-2V4zm2 6a1 1 0 011-1h6a1 1 0 110 2H7a1 1 0 01-1-1zm1 3a1 1 0 100 2h6a1 1 0 100-2H7z" clipRule="evenodd" />
        </svg>
        Document Text Extraction
        <span className="ml-2 text-xs bg-[#7C3AED]/20 text-[#7C3AED] px-2 py-1 rounded-full">PDF & EML</span>
      </h2>

      <div className="space-y-4">
        {!file && !result ? (
          <div 
            className={`flex flex-col items-center justify-center p-8 border-2 rounded-xl transition-all duration-200 ${
              dragActive 
                ? 'border-[#FF8C66] bg-[#FF8C66]/10 scale-105' 
                : 'border-dashed border-theme-secondary bg-[var(--bg-input)] hover:bg-[var(--bg-hover)]'
            }`}
            onDrop={handleDrop}
            onDragOver={handleDrag}
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
          >
            <div className="relative">
              <svg xmlns="http://www.w3.org/2000/svg" className="w-12 h-12 text-theme-muted mb-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
              </svg>
              {dragActive && (
                <div className="absolute inset-0 flex items-center justify-center bg-[#FF8C66]/20 rounded-full">
                  <svg xmlns="http://www.w3.org/2000/svg" className="w-8 h-8 text-[#FF8C66]" viewBox="0 0 20 20" fill="currentColor">
                    <path fillRule="evenodd" d="M16.707 10.293a1 1 0 010 1.414l-6 6a1 1 0 01-1.414 0l-6-6a1 1 0 111.414-1.414L9 14.586V3a1 1 0 012 0v11.586l4.293-4.293a1 1 0 011.414 0z" clipRule="evenodd" />
                  </svg>
                </div>
              )}
            </div>
            <p className="text-theme-secondary mb-2 font-medium">Drag & drop your PDF or EML file here</p>
            <p className="text-theme-muted text-sm mb-1">or</p>
            <button
              onClick={triggerFileSelect}
              className="px-6 py-3 bg-[#FF8C66] hover:bg-[#ff7a4d] text-black rounded-lg transition-all duration-200 font-medium shadow-lg shadow-[#FF8C66]/20 hover:shadow-[#FF8C66]/30 transform hover:-translate-y-0.5"
            >
              Browse Files
            </button>
            <p className="text-theme-muted text-xs mt-4">Supports PDF and EML files up to 50MB</p>
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleFileChange}
              accept=".pdf,application/pdf,.eml,message/rfc822"
              className="hidden"
            />
          </div>
        ) : (
          <div className="space-y-4">
            {file && (
              <div className="p-4 bg-[var(--bg-input)] rounded-xl border border-theme">
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center">
                    <svg xmlns="http://www.w3.org/2000/svg" className="w-5 h-5 mr-2 text-[#7C3AED]" viewBox="0 0 20 20" fill="currentColor">
                      <path fillRule="evenodd" d="M4 4a2 2 0 012-2h4.586A2 2 0 0112 2.586L15.414 6A2 2 0 0116 7.414V16a2 2 0 01-2 2H6a2 2 0 01-2-2V4zm2 6a1 1 0 011-1h6a1 1 0 110 2H7a1 1 0 01-1-1zm1 3a1 1 0 100 2h6a1 1 0 100-2H7z" clipRule="evenodd" />
                    </svg>
                    <div>
                      <span className="text-sm font-mono truncate max-w-[180px] block text-heading">{file.name}</span>
                      <span className="text-xs text-theme-muted">{formatFileSize(file.size)}</span>
                    </div>
                  </div>
                  <button
                    onClick={handleReset}
                    className="text-theme-muted hover:text-theme-primary transition-colors"
                  >
                    <svg xmlns="http://www.w3.org/2000/svg" className="w-5 h-5" viewBox="0 0 20 20" fill="currentColor">
                      <path fillRule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clipRule="evenodd" />
                    </svg>
                  </button>
                </div>
                
                {isLoading && (
                  <div className="mb-4">
                    <div className="flex justify-between text-sm text-theme-muted mb-1">
                      <span>Processing document...</span>
                      <span>{progress}%</span>
                    </div>
                    <div className="w-full bg-[var(--bg-input)] rounded-full h-2">
                      <div 
                        className="bg-[#FF8C66] h-2 rounded-full transition-all duration-300"
                        style={{ width: `${progress}%` }}
                      ></div>
                    </div>
                  </div>
                )}
                
                <div className="flex justify-center">
                  <button
                    onClick={handleExtractText}
                    disabled={isLoading}
                    className={`px-6 py-2.5 rounded-lg transition-all duration-200 flex items-center font-medium ${
                      isLoading
                        ? 'bg-[var(--bg-hover)] cursor-not-allowed text-theme-muted'
                        : 'bg-[#FF8C66] hover:bg-[#ff7a4d] text-black shadow-lg shadow-[#FF8C66]/20 hover:shadow-[#FF8C66]/30 transform hover:-translate-y-0.5'
                    }`}
                  >
                    {isLoading ? (
                      <>
                        <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                        </svg>
                        Processing...
                      </>
                    ) : (
                      <>
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2" viewBox="0 0 20 20" fill="currentColor">
                          <path fillRule="evenodd" d="M3 17a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zm3.293-7.707a1 1 0 011.414 0L9 10.586V3a1 1 0 112 0v7.586l1.293-1.293a1 1 0 111.414 1.414l-3 3a1 1 0 01-1.414 0l-3-3a1 1 0 010-1.414z" clipRule="evenodd" />
                        </svg>
                        Extract Text
                      </>
                    )}
                  </button>
                </div>
              </div>
            )}

            {error && (
              <div className="p-4 bg-red-500/10 border border-red-500/30 rounded-lg animate-fadeIn">
                <div className="flex items-center text-red-400">
                  <svg xmlns="http://www.w3.org/2000/svg" className="w-5 h-5 mr-2" viewBox="0 0 20 20" fill="currentColor">
                    <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
                  </svg>
                  <span className="font-medium">Error</span>
                </div>
                <p className="mt-1 text-sm text-red-400">{error}</p>
              </div>
            )}

            {result && (
              <div className="space-y-4 animate-fadeIn">
                <div className="p-4 bg-green-500/10 border border-green-500/30 rounded-lg">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center text-green-400">
                      <svg xmlns="http://www.w3.org/2000/svg" className="w-5 h-5 mr-2" viewBox="0 0 20 20" fill="currentColor">
                        <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                      </svg>
                      <span className="font-medium">Extraction Successful</span>
                    </div>
                    <button
                      onClick={handleClearResult}
                      className="text-theme-muted hover:text-theme-primary transition-colors"
                    >
                      <svg xmlns="http://www.w3.org/2000/svg" className="w-5 h-5" viewBox="0 0 20 20" fill="currentColor">
                        <path fillRule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clipRule="evenodd" />
                      </svg>
                    </button>
                  </div>
                  <p className="mt-2 text-sm text-green-400">File: {result.filename}</p>
                  {result.message && (
                    <p className="mt-1 text-sm text-yellow-400 italic">{result.message}</p>
                  )}

                  {/* Generate 3D from document button */}
                  {file && onModelGenerated && (
                    <div className="mt-4">
                      <button
                        onClick={handleGenerate3D}
                        disabled={generating3D}
                        className={`w-full px-4 py-2.5 rounded-lg font-bold transition-all flex items-center justify-center gap-2 ${
                          generating3D
                            ? 'bg-[#7C3AED]/40 text-white/60 cursor-not-allowed'
                            : 'bg-[#7C3AED] hover:bg-[#6d28d9] text-white shadow-lg shadow-[#7C3AED]/20'
                        }`}
                      >
                        {generating3D ? (
                          <>
                            <svg className="animate-spin h-4 w-4" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                            </svg>
                            Generating 3D model from document…
                          </>
                        ) : (
                          <>
                            <svg xmlns="http://www.w3.org/2000/svg" className="w-4 h-4" viewBox="0 0 20 20" fill="currentColor">
                              <path d="M10 12a2 2 0 100-4 2 2 0 000 4z" />
                              <path fillRule="evenodd" d="M.458 10C1.732 5.943 5.522 3 10 3s8.268 2.943 9.542 7c-1.274 4.057-5.064 7-9.542 7S1.732 14.057.458 10zM14 10a4 4 0 11-8 0 4 4 0 018 0z" clipRule="evenodd" />
                            </svg>
                            Generate 3D from this document
                          </>
                        )}
                      </button>
                      {gen3DError && (
                        <p className="mt-2 text-xs text-red-400">{gen3DError}</p>
                      )}
                    </div>
                  )}
                </div>

                {result.extracted_text && (
                  <div className="space-y-3">
                    <div className="flex justify-between items-center">
                      <div className="flex items-center">
                        <svg xmlns="http://www.w3.org/2000/svg" className="w-5 h-5 mr-2 text-[#7C3AED]" viewBox="0 0 20 20" fill="currentColor">
                          <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
                        </svg>
                        <h3 className="font-medium text-heading">
                          Extracted Content
                        </h3>
                        <span className="ml-3 text-xs bg-[var(--bg-input)] text-theme-muted px-2 py-1 rounded">
                          {getTextStats(result.extracted_text).words} words
                        </span>
                        <span className="ml-2 text-xs bg-[var(--bg-input)] text-theme-muted px-2 py-1 rounded">
                          {getTextStats(result.extracted_text).lines} lines
                        </span>
                      </div>
                      <div className="flex gap-2">
                        <button
                          onClick={() => navigator.clipboard.writeText(result.extracted_text || '')}
                          className="text-xs px-3 py-1.5 bg-[var(--bg-input)] hover:bg-[var(--bg-hover)] rounded text-theme-secondary flex items-center transition-colors"
                        >
                          <svg xmlns="http://www.w3.org/2000/svg" className="w-3 h-3 mr-1" viewBox="0 0 20 20" fill="currentColor">
                            <path d="M8 3a1 1 0 011-1h2a1 1 0 110 2H9a1 1 0 01-1-1z" />
                            <path d="M6 3a2 2 0 00-2 2v11a2 2 0 002 2h8a2 2 0 002-2V5a2 2 0 00-2-2 3 3 0 01-3 3H9a3 3 0 01-3-3z" />
                          </svg>
                          Copy Text
                        </button>
                        <button
                          onClick={() => {
                            const blob = new Blob([result.extracted_text], { type: 'text/plain' });
                            const url = URL.createObjectURL(blob);
                            const a = document.createElement('a');
                            a.href = url;
                            a.download = `${result.filename.replace(/\.(pdf|eml)$/i, '')}_extracted.txt`;
                            document.body.appendChild(a);
                            a.click();
                            document.body.removeChild(a);
                            URL.revokeObjectURL(url);
                          }}
                          className="text-xs px-3 py-1.5 bg-[var(--bg-input)] hover:bg-[var(--bg-hover)] rounded text-theme-secondary flex items-center transition-colors"
                        >
                          <svg xmlns="http://www.w3.org/2000/svg" className="w-3 h-3 mr-1" viewBox="0 0 20 20" fill="currentColor">
                            <path fillRule="evenodd" d="M3 17a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zm3.293-7.707a1 1 0 011.414 0L9 10.586V3a1 1 0 112 0v7.586l1.293-1.293a1 1 0 111.414 1.414l-3 3a1 1 0 01-1.414 0l-3-3a1 1 0 010-1.414z" clipRule="evenodd" />
                          </svg>
                          Download
                        </button>
                      </div>
                    </div>
                    <div className="bg-[var(--bg-tertiary)] p-4 rounded-xl border border-theme max-h-60 overflow-y-auto">
                      <pre className="whitespace-pre-wrap text-xs text-theme-secondary font-mono leading-relaxed">
                        {result.extracted_text}
                      </pre>
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        )}
      </div>
      
      {/* Toast Notifications */}
      <div className="fixed top-4 right-4 z-50 space-y-2">
        {toasts.map((toast) => (
          <div
            key={toast.id}
            className={`
              flex items-center p-4 rounded-lg shadow-lg transform transition-all duration-300 ease-in-out
              ${toast.type === 'success' ? 'bg-green-500/90 text-white border border-green-400/50' : ''}
              ${toast.type === 'error' ? 'bg-red-500/90 text-white border border-red-400/50' : ''}
              ${toast.type === 'warning' ? 'bg-yellow-500/90 text-white border border-yellow-400/50' : ''}
              ${toast.type === 'info' ? 'bg-blue-500/90 text-white border border-blue-400/50' : ''}
              animate-fadeIn
            `}
          >
            <div className="flex-1 flex items-center">
              {toast.type === 'success' && (
                <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                </svg>
              )}
              {toast.type === 'error' && (
                <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                </svg>
              )}
              {toast.type === 'warning' && (
                <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
                </svg>
              )}
              {toast.type === 'info' && (
                <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
                </svg>
              )}
              <span className="text-sm font-medium">{toast.message}</span>
            </div>
            <button
              onClick={() => removeToast(toast.id)}
              className="ml-2 text-white/70 hover:text-white transition-colors"
            >
              <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" viewBox="0 0 20 20" fill="currentColor">
                <path fillRule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clipRule="evenodd" />
              </svg>
            </button>
          </div>
        ))}
      </div>
    </div>
  );
};
