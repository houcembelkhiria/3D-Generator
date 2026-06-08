import React from 'react';
import { Modal } from './Modal';
import { IconFileText, IconTrash } from './Icons';

interface TextExtractionResult {
  filename: string;
  file_type?: string;
  extracted_text?: string;
  message?: string;
}

interface ExtractionHistoryModalProps {
  isOpen: boolean;
  onClose: () => void;
  results: TextExtractionResult[];
  onClear: () => void;
}

export const ExtractionHistoryModal: React.FC<ExtractionHistoryModalProps> = ({ 
  isOpen, 
  onClose, 
  results, 
  onClear 
}) => {
  return (
    <Modal 
      isOpen={isOpen} 
      onClose={onClose} 
      title="Extraction History" 
      size="lg"
    >
      <div className="space-y-4">
        {results.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-12 text-theme-muted">
            <IconFileText className="w-12 h-12 mb-4 opacity-20" />
            <p className="text-lg font-medium">No files processed yet</p>
            <p className="text-sm mt-1">Upload a file to see extraction results</p>
          </div>
        ) : (
          <>
            <div className="flex justify-between items-center">
              <p className="text-theme-muted">
                Showing {results.length} extraction{results.length !== 1 ? 's' : ''}
              </p>
              <button
                onClick={onClear}
                className="flex items-center gap-2 px-3 py-2 bg-red-500/20 hover:bg-red-500/30 text-red-400 rounded-lg text-sm font-medium transition-colors border border-red-500/30"
              >
                <IconTrash className="w-4 h-4" />
                Clear All
              </button>
            </div>
            
            <div className="space-y-3 max-h-96 overflow-y-auto pr-2 scrollbar-hide">
              {results.map((result, index) => (
                <div 
                  key={index} 
                  className="bg-[var(--bg-tertiary)] p-4 rounded-xl border border-theme hover:border-[var(--border-secondary)] transition-colors"
                >
                  <div className="flex justify-between items-start mb-3">
                    <h4 className="font-medium text-heading truncate flex-1 mr-3">
                      {result.filename}
                    </h4>
                    <span className="text-xs bg-[#7C3AED]/20 text-[#7C3AED] px-2 py-1 rounded whitespace-nowrap">
                      {result.file_type || 'Unknown'}
                    </span>
                  </div>
                  
                  {result.extracted_text && (
                    <p className="text-theme-muted text-sm mb-3 line-clamp-3 bg-[var(--bg-input)] p-3 rounded-lg">
                      {result.extracted_text.substring(0, 200)}
                      {result.extracted_text.length > 200 ? '...' : ''}
                    </p>
                  )}
                  
                  {result.message && (
                    <div className="text-amber-400 text-xs bg-amber-500/10 border border-amber-500/20 rounded-lg p-2 mb-3">
                      {result.message}
                    </div>
                  )}
                  
                  <div className="flex justify-between text-xs text-theme-muted">
                    <span>
                      {result.extracted_text?.length || 0} characters
                    </span>
                    <span>
                      Entry #{results.length - index}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </>
        )}
      </div>
    </Modal>
  );
};
