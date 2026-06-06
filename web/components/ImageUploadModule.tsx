"use client";

import { useState, ChangeEvent } from "react";
import { api } from "@/services/api";

// Atualizado para refletir o JSON real da API
export interface AnalysisResult {
  faceShape: unknown;
  message: string;
  recommendation: {
    id: number;
    faceShape: string;
    confidence: number;
    description: string;
    characteristics: string[];
    justification: string;
    suggestedCuts: { nome: string; justificativa: string }[]; // ← corrigido
    avoid: string[];
  };
  metrics: {
    face_height_px: number;
    face_width_px: number;
    ratio_height_width: number;
    ratio_jaw_cheek: number;
    ratio_forehead_jaw: number;
  };
  photoUrl: string;
}
interface ImageUploadModuleProps {
  clientId: string;
  onAnalysisComplete?: (result: AnalysisResult) => void;
}

export default function ImageUploadModule({ clientId, onAnalysisComplete }: ImageUploadModuleProps) {
  const [isUploading, setIsUploading] = useState(false);
  const [preview, setPreview] = useState<string | null>(null);

  const handleFileChange = async (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    // Gerar pré-visualização local
    const reader = new FileReader();
    reader.onloadend = () => setPreview(reader.result as string);
    reader.readAsDataURL(file);

    setIsUploading(true);

    const formData = new FormData();
    formData.append("file", file);
    formData.append("clientId", clientId);

    try {
      const response = await api.post("/visagismo/analisar", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      if (onAnalysisComplete) {
        onAnalysisComplete(response.data);
      }
    } catch (error) {
      console.error("Erro ao processar imagem pela IA:", error);
      alert("Erro ao realizar análise. Tente novamente.");
      window.location.reload();
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div className="flex flex-col items-center justify-center p-6 border-2 border-dashed border-outline-variant/30 rounded-2xl bg-surface-container-low hover:border-primary/50 transition-all">
      {preview ? (
        <div className="relative w-32 h-32 mb-4">
          <img src={preview} alt="Preview" className="w-full h-full object-cover rounded-xl" />
        </div>
      ) : (
        <span className="material-symbols-outlined text-4xl text-on-surface-variant mb-4">add_a_photo</span>
      )}

      <label className="cursor-pointer bg-primary text-black px-4 py-2 rounded-lg font-label text-xs uppercase tracking-wider hover:bg-primary/90">
        {isUploading ? "Analisando..." : "Selecionar Foto"}
        <input type="file" className="hidden" accept="image/*" onChange={handleFileChange} disabled={isUploading} />
      </label>

      {isUploading && <p className="mt-2 text-xs text-primary animate-pulse">IA está processando o perfil...</p>}
    </div>
  );
}
