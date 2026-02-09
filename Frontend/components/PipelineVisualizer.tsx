import React from 'react';
import { PipelineStep, GenerationMethod } from '../types';
import { IconFileText, IconBrain, IconBox, IconActivity, IconCheckCircle, IconLoader } from './Icons';

interface PipelineVisualizerProps {
  currentStep: PipelineStep;
  generationMethod?: GenerationMethod;
}

interface StepItemProps {
  step: PipelineStep;
  title: string;
  description: string;
  isActive: boolean;
  isCompleted: boolean;
  icon: React.ElementType;
}

const StepItem: React.FC<StepItemProps> = ({ 
  step, 
  title, 
  description, 
  isActive, 
  isCompleted, 
  icon: Icon 
}) => {
  return (
    <div className={`relative flex items-center p-4 rounded-xl border transition-all duration-300 ${
      isActive 
        ? 'bg-[#FF8C66]/10 border-[#FF8C66] shadow-lg shadow-[#FF8C66]/10' 
        : isCompleted 
          ? 'bg-zinc-900/80 border-[#7C3AED]/50' 
          : 'bg-[#0A0A0A] border-zinc-800 opacity-60'
    }`}>
      <div className={`flex items-center justify-center w-12 h-12 rounded-lg mr-4 ${
        isActive 
          ? 'bg-[#FF8C66] text-black animate-pulse' 
          : isCompleted 
            ? 'bg-[#7C3AED] text-white' 
            : 'bg-zinc-800 text-zinc-500'
      }`}>
        <Icon className="w-6 h-6" />
      </div>
      <div className="flex-1">
        <h3 className={`font-bold ${isActive ? 'text-[#FF8C66]' : isCompleted ? 'text-[#7C3AED]' : 'text-zinc-400'}`}>
          {title}
        </h3>
        <p className="text-xs text-zinc-500 mt-1">{description}</p>
      </div>
      <div className="ml-2">
        {isActive && <IconLoader className="w-5 h-5 text-[#FF8C66] animate-spin" />}
        {isCompleted && <IconCheckCircle className="w-5 h-5 text-[#7C3AED]" />}
      </div>
    </div>
  );
};

export const PipelineVisualizer: React.FC<PipelineVisualizerProps> = ({ currentStep, generationMethod }) => {
  const steps = [
    {
      id: PipelineStep.INGESTION,
      title: 'ÉTAPE A : INGESTION',
      description: 'Lib unstructured. Nettoyage bruit & parsing (PDF/EML).',
      icon: IconFileText
    },
    {
      id: PipelineStep.EXTRACTION,
      title: 'ÉTAPE B : CERVEAU LLM',
      description: 'Llama 3 Local (vLLM). Extraction JSON via Pydantic.',
      icon: IconBrain
    },
    {
      id: PipelineStep.GENERATION,
      title: 'ÉTAPE C : GÉNÉRATION 3D',
      description: generationMethod === GenerationMethod.PROCEDURAL 
        ? 'Méthode Procédurale: Qwen 2.5 Coder -> C# Script'
        : 'Méthode Visuelle: TripoSR -> .GLB',
      icon: IconBox
    },
    {
      id: PipelineStep.MCP_DISPATCH,
      title: 'ÉTAPE D : AGENT MCP',
      description: 'Pilotage Unity Editor. call_tool("SpawnAsset").',
      icon: IconActivity
    }
  ];

  const getStepStatus = (stepId: PipelineStep) => {
    const stepOrder = [
      PipelineStep.IDLE,
      PipelineStep.INGESTION,
      PipelineStep.EXTRACTION,
      PipelineStep.GENERATION,
      PipelineStep.MCP_DISPATCH,
      PipelineStep.COMPLETED
    ];
    
    const currentIndex = stepOrder.indexOf(currentStep);
    const stepIndex = stepOrder.indexOf(stepId);

    return {
      isActive: currentStep === stepId,
      isCompleted: currentIndex > stepIndex
    };
  };

  return (
    <div className="space-y-4">
      {steps.map((step) => {
        const { isActive, isCompleted } = getStepStatus(step.id);
        return (
          <StepItem
            key={step.id}
            step={step.id}
            title={step.title}
            description={step.description}
            isActive={isActive}
            isCompleted={isCompleted}
            icon={step.icon}
          />
        );
      })}
    </div>
  );
};