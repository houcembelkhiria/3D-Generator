import React from 'react';
import { useTheme } from './ThemeContext';
import { IconSun, IconMoon } from './Icons';

export const ThemeToggle: React.FC = () => {
  const { theme, toggleTheme } = useTheme();

  return (
    <button
      onClick={toggleTheme}
      className="p-2 rounded-lg bg-theme-input hover:bg-[var(--bg-hover)] text-theme-muted hover:text-theme-primary transition-all duration-200 flex items-center justify-center border border-theme"
      aria-label={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}
    >
      {theme === 'dark' ? (
        <IconSun className="w-5 h-5" />
      ) : (
        <IconMoon className="w-5 h-5" />
      )}
    </button>
  );
};
