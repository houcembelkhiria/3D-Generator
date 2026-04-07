import React from 'react';
import { IconLayoutDashboard, IconSettings, IconFileText } from './Icons';

interface SidebarProps {
  activeView: 'agent' | 'files' | 'settings';
  onViewChange: (view: 'agent' | 'files' | 'settings') => void;
}

export const Sidebar: React.FC<SidebarProps> = ({ activeView, onViewChange }) => {
  return (
    <div className="w-20 lg:w-64 sidebar flex flex-col items-center lg:items-stretch py-6 space-y-2 flex-shrink-0">
      <div className="flex items-center justify-center lg:justify-start lg:px-6 mb-8">
         <div className="w-8 h-8 bg-gradient-to-br from-[#FF8C66] to-[#7C3AED] rounded-lg flex items-center justify-center text-white font-bold text-xl shadow-lg shadow-[#FF8C66]/20">
            M
         </div>
         <span className="hidden lg:block ml-3 font-bold text-lg text-theme-primary tracking-tight">MCP Agent</span>
      </div>

      <button
        onClick={() => onViewChange('agent')}
        className={`w-full lg:w-auto mx-2 lg:mx-4 p-3 lg:px-4 lg:py-3 flex items-center justify-center lg:justify-start sidebar-item ${
          activeView === 'agent'
            ? 'active'
            : ''
        }`}
      >
        <IconLayoutDashboard className={`w-6 h-6 ${activeView === 'agent' ? 'text-white' : 'text-theme-muted'}`} />
        <span className="hidden lg:block ml-3 font-medium text-theme-primary">Agent</span>
      </button>

      <button
        onClick={() => onViewChange('files')}
        className={`w-full lg:w-auto mx-2 lg:mx-4 p-3 lg:px-4 lg:py-3 flex items-center justify-center lg:justify-start sidebar-item ${
          activeView === 'files'
            ? 'active'
            : ''
        }`}
      >
        <IconFileText className={`w-6 h-6 ${activeView === 'files' ? 'text-white' : 'text-theme-muted'}`} />
        <span className="hidden lg:block ml-3 font-medium text-theme-primary">Files Treatment</span>
      </button>

      <button
        onClick={() => onViewChange('settings')}
        className={`w-full lg:w-auto mx-2 lg:mx-4 p-3 lg:px-4 lg:py-3 flex items-center justify-center lg:justify-start sidebar-item ${
          activeView === 'settings'
            ? 'active'
            : ''
        }`}
      >
        <IconSettings className={`w-6 h-6 ${activeView === 'settings' ? 'text-white' : 'text-theme-muted'}`} />
        <span className="hidden lg:block ml-3 font-medium text-theme-primary">System Config</span>
      </button>
    </div>
  );
}