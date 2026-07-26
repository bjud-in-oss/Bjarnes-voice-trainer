import React from 'react';
import BjarneVoiceTrainer from './components/BjarneVoiceTrainer';

const App: React.FC = () => {
  return (
    <div className="min-h-screen bg-slate-900 text-white font-sans selection:bg-blue-500 selection:text-white">
      <header className="p-6 border-b border-slate-700 bg-slate-800/50 sticky top-0 z-50 backdrop-blur-md">
        <div className="max-w-5xl mx-auto flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-emerald-400">
              Bjarne's Voice Trainer
            </h1>
            <p className="text-slate-400 text-sm">Audio Classification System v1.0</p>
          </div>
          <div className="text-xs text-slate-500 font-mono hidden sm:block">
            TENSORFLOW.JS // SPEECH-COMMANDS
          </div>
        </div>
      </header>

      <main className="p-4 sm:p-6 max-w-5xl mx-auto">
        <BjarneVoiceTrainer />
      </main>
      
      <footer className="p-6 text-center text-slate-600 text-xs mt-12">
        Designed for Aphasia Accessibility Support
      </footer>
    </div>
  );
};

export default App;