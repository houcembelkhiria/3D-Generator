import React, { useState, useCallback } from 'react';
import { TextExtractor } from './TextExtractor';
import { ExtractionHistoryModal } from './ExtractionHistoryModal';

interface TextExtractionResult {
  filename: string;
  file_type?: string;
  extracted_text?: string;
  message?: string;
}

interface FilesTreatmentProps {
  onTextExtracted?: (text: string) => void;
}

export const FilesTreatment: React.FC<FilesTreatmentProps> = ({ onTextExtracted }) => {
  const [extractedResults, setExtractedResults] = useState<TextExtractionResult[]>([]);
  const [isHistoryModalOpen, setIsHistoryModalOpen] = useState(false);
  
  const handleExtractComplete = useCallback((result: TextExtractionResult) => {
    setExtractedResults(prev => [...prev, result]);
    if (onTextExtracted) {
      onTextExtracted(result.extracted_text || '');
    }
  }, [onTextExtracted]);

  const clearResults = () => {
    setExtractedResults([]);
  };

  return (
    <div className="space-y-6">
      <div className="bg-[var(--bg-card)] p-6 rounded-2xl border border-theme">
        <div className="flex justify-between items-center mb-6">
          <h2 className="text-2xl font-bold text-heading flex items-center">
            <svg xmlns="http://www.w3.org/2000/svg" className="w-7 h-7 mr-3 text-[#7C3AED]" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z"/>
              <polyline points="14 2 14 8 20 8"/>
              <line x1="16" y1="13" x2="8" y2="13"/>
              <line x1="16" y1="17" x2="8" y2="17"/>
              <line x1="10" y1="9" x2="8" y2="9"/>
            </svg>
            Files Treatment
          </h2>
          <div className="flex gap-3">
            {extractedResults.length > 0 && (
              <button
                onClick={clearResults}
                className="px-4 py-2 bg-red-500/20 hover:bg-red-500/30 text-red-400 rounded-lg text-sm font-medium transition-colors border border-red-500/30 flex items-center gap-2"
              >
                <svg xmlns="http://www.w3.org/2000/svg" className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M3 6h18"/><path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6"/><path d="M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2"/><line x1="10" y1="11" x2="10" y2="17"/><line x1="14" y1="11" x2="14" y2="17"/>
                </svg>
                Clear Results
              </button>
            )}
            <button
              onClick={() => setIsHistoryModalOpen(true)}
              className="px-4 py-2 bg-[#7C3AED] hover:bg-[#6d28d9] text-white rounded-lg text-sm font-medium transition-colors flex items-center gap-2"
            >
              <svg xmlns="http://www.w3.org/2000/svg" className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/>
              </svg>
              View History ({extractedResults.length})
            </button>
          </div>
        </div>
        
        <p className="text-body mb-6">
          Upload PDF or EML files for automatic text extraction. The system will detect file types automatically and extract content for use in asset generation.
        </p>
        
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div>
            <TextExtractor onExtractComplete={handleExtractComplete} />
          </div>
        </div>
      </div>
      
      {/* Extraction History Modal */}
      <ExtractionHistoryModal
        isOpen={isHistoryModalOpen}
        onClose={() => setIsHistoryModalOpen(false)}
        results={extractedResults}
        onClear={clearResults}
      />
    </div>
  );
};