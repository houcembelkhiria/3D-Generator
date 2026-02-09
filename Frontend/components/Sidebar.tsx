import React from 'react';
import { IconLayoutDashboard, IconSettings } from './Icons';

interface SidebarProps {
  activeView: 'dashboard' | 'settings';
  onViewChange: (view: 'dashboard' | 'settings') => void;
}

export const Sidebar: React.FC<SidebarProps> = ({ activeView, onViewChange }) => {
  return (
    <div className="w-20 lg:w-64 bg-[#050505] border-r border-zinc-800 flex flex-col items-center lg:items-stretch py-6 space-y-2 flex-shrink-0">
      <div className="flex items-center justify-center lg:justify-start lg:px-6 mb-8">
         <div className="w-8 h-8 bg-gradient-to-br from-[#FF8C66] to-[#7C3AED] rounded-lg flex items-center justify-center text-white font-bold text-xl shadow-lg shadow-[#FF8C66]/20">
            M
         </div>
         <span className="hidden lg:block ml-3 font-bold text-lg text-zinc-200 tracking-tight">MCP Agent</span>
      </div>

      <button
        onClick={() => onViewChange('dashboard')}
        className={`w-full lg:w-auto mx-2 lg:mx-4 p-3 lg:px-4 lg:py-3 rounded-xl flex items-center justify-center lg:justify-start transition-all duration-200 group ${
          activeView === 'dashboard'
            ? 'bg-[#FF8C66] text-black shadow-lg shadow-[#FF8C66]/20'
            : 'text-zinc-500 hover:bg-zinc-900 hover:text-zinc-200'
        }`}
      >
        <IconLayoutDashboard className={`w-6 h-6 ${activeView === 'dashboard' ? 'text-black' : 'text-zinc-500 group-hover:text-zinc-200'}`} />
        <span className="hidden lg:block ml-3 font-medium">Dashboard</span>
      </button>

      <button
        onClick={() => onViewChange('settings')}
        className={`w-full lg:w-auto mx-2 lg:mx-4 p-3 lg:px-4 lg:py-3 rounded-xl flex items-center justify-center lg:justify-start transition-all duration-200 group ${
          activeView === 'settings'
            ? 'bg-[#FF8C66] text-black shadow-lg shadow-[#FF8C66]/20'
            : 'text-zinc-500 hover:bg-zinc-900 hover:text-zinc-200'
        }`}
      >
        <IconSettings className={`w-6 h-6 ${activeView === 'settings' ? 'text-black' : 'text-zinc-500 group-hover:text-zinc-200'}`} />
        <span className="hidden lg:block ml-3 font-medium">System Config</span>
      </button>
    </div>
  );
}