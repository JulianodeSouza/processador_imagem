/* eslint-disable @next/next/no-img-element */
import { AnalysisResult } from "./ImageUploadModule";

interface VisagismResultProps {
    result: AnalysisResult | null;
}

export default function VisagismResult({ result }: VisagismResultProps) {
    // Se o resultado for nulo (análise ainda não foi feita ou deu erro), não renderiza nada.
    if (!result) return null;

    const { recommendation, photoUrl } = result;

    return (
        <div className="mt-8 p-6 bg-white border border-gray-200 rounded-2xl shadow-sm flex flex-col md:flex-row gap-8 animate-in fade-in slide-in-from-bottom-4 duration-500">

            {/* Coluna da Esquerda: Foto Analisada e Confiança */}
            <div className="flex-shrink-0 flex flex-col items-center">
                {photoUrl ? (
                    <img
                        src={photoUrl}
                        alt="Foto Analisada"
                        className="w-48 h-48 object-cover rounded-xl shadow-md border border-gray-100"
                    />
                ) : (
                    <div className="w-48 h-48 bg-surface-container-high rounded-xl shadow-md border border-outline-variant/30 flex items-center justify-center text-on-surface-variant">
                        <span className="material-symbols-outlined text-6xl opacity-50">face</span>
                    </div>
                )}
            </div>

            {/* Coluna da Direita: Dados da Análise */}
            <div className="flex-grow">
                <h2 className="text-2xl font-bold text-gray-800 mb-2">Diagnóstico de Visagismo</h2>
                {/* <p className="text-gray-600 mb-6">{recommendation.message}</p> */}

                {/* Formato do Rosto e Descrição */}
                <div className="bg-gray-50 rounded-xl p-4 mb-6">
                    <p className="text-sm text-gray-500 uppercase tracking-wide font-semibold mb-1">Formato Identificado</p>
                    <p className="text-xl font-bold text-primary mb-2">{recommendation.faceShape}</p>
                    <p className="text-gray-700 text-sm leading-relaxed">
                        {recommendation.description}
                    </p>
                    <p className="text-gray-800 text-sm font-medium mt-3 bg-white p-3 rounded border border-gray-200">
                        💡 <strong>Dica de Ouro:</strong> {recommendation.justification}
                    </p>
                </div>

                {/* Cortes Sugeridos */}
                <div className="mb-6">
                    <h3 className="text-lg font-bold text-gray-800 border-b pb-2 mb-3">Cortes Recomendados</h3>
                    <ul className="space-y-3">
                        {recommendation.suggestedCuts.map((corte, index) => (
                            <li key={index} className="bg-green-50/50 border border-green-100 p-3 rounded-lg flex flex-col">
                                <span className="font-bold text-green-800">{corte.nome}</span>
                                <span className="text-sm text-green-700 mt-1">{corte.justificativa}</span>
                            </li>
                        ))}
                    </ul>
                </div>

                {/* O que evitar */}
                {recommendation.avoid && recommendation.avoid.length > 0 && (
                    <div className="mb-6">
                        <h3 className="text-lg font-bold text-gray-800 border-b pb-2 mb-3">O que Evitar</h3>
                        <ul className="list-disc list-inside space-y-1">
                            {recommendation.avoid.map((item, index) => (
                                <li key={index} className="text-sm text-red-600 font-medium">{item}</li>
                            ))}
                        </ul>
                    </div>
                )}

                {/* Características Detectadas (Tags) */}
                {recommendation.characteristics && recommendation.characteristics.length > 0 && (
                    <div>
                        <h3 className="text-sm font-bold text-gray-500 uppercase tracking-wide mb-3">Características Observadas</h3>
                        <div className="flex flex-wrap gap-2">
                            {recommendation.characteristics.map((char, index) => (
                                <span key={index} className="bg-gray-100 text-gray-600 px-3 py-1 rounded-full text-xs font-medium">
                                    {char}
                                </span>
                            ))}
                        </div>
                    </div>
                )}

            </div>
        </div>
    );
}