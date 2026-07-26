import React, { useState, useEffect, useRef, useCallback } from 'react';
import * as tf from '@tensorflow/tfjs';
import * as speechCommands from '@tensorflow-models/speech-commands';
import JSZip from 'jszip';

// --- Constants ---
const LABELS = {
  NOISE: 'Bakgrundsbrus',
  YES: 'Hm-hm (Ja)',
};

const LABEL_KEYS = Object.keys(LABELS) as Array<keyof typeof LABELS>;
const THRESHOLD = 0.85;

interface ClassData {
  name: string;
  id: string;
  color: string;
  icon: string;
}

const INITIAL_CLASSES: ClassData[] = [
  { id: '0', name: LABELS.NOISE, color: 'bg-slate-600', icon: '🔇' },
  { id: '1', name: LABELS.YES, color: 'bg-emerald-600', icon: '👍' },
];

interface AudioSample {
    id: string; // Unique ID for React key
    tfUid: string; // UID inside TensorFlow dataset
    classId: string;
    blob: Blob;
    url: string;
    timestamp: number;
}

// --- Helpers ---
const simpleUUID = () => {
    // Fallback for environments where crypto.randomUUID is unavailable (e.g. non-HTTPS)
    // @ts-ignore
    if (typeof crypto !== 'undefined' && crypto.randomUUID) {
        // @ts-ignore
        return crypto.randomUUID();
    }
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
        var r = Math.random() * 16 | 0, v = c === 'x' ? r : (r & 0x3 | 0x8);
        return v.toString(16);
    });
};

const BjarneVoiceTrainer: React.FC = () => {
  // --- State ---
  const [isModelLoading, setIsModelLoading] = useState(true);
  const [isTraining, setIsTraining] = useState(false);
  const [isListening, setIsListening] = useState(false);
  
  const isTrainedRef = useRef(false);
  const [isTrained, setIsTrained] = useState(false); 

  const [trainingProgress, setTrainingProgress] = useState<{ loss: number; accuracy: number } | null>(null);
  
  // Samples Management
  const [samples, setSamples] = useState<AudioSample[]>([]);

  // Inference results
  const [probabilities, setProbabilities] = useState<number[]>([0, 0]);
  const [detectedLabel, setDetectedLabel] = useState<string | null>(null);

  // Hidden file input refs
  const zipInputRef = useRef<HTMLInputElement>(null);

  // --- Refs for TFJS objects ---
  const recognizerRef = useRef<speechCommands.SpeechCommandRecognizer | null>(null);
  const transferRecognizerRef = useRef<speechCommands.TransferSpeechCommandRecognizer | null>(null);
  
  // --- Logic Locks ---
  const isCollectingRef = useRef<boolean>(false);
  const isRecordingRef = useRef<boolean>(false); // Mutex lock to prevent parallel recordings
  const currentSessionId = useRef<number>(0); // Tracks the current "press" session

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);

  // --- Initialization ---
  useEffect(() => {
    const loadModel = async () => {
      try {
        console.log('Initializing TensorFlow.js...');
        await tf.ready();
        
        if (!tf.getBackend()) {
            try {
                await tf.setBackend('webgl');
            } catch (e) {
                await tf.setBackend('cpu');
            }
        }

        const recognizer = speechCommands.create('BROWSER_FFT');
        await recognizer.ensureModelLoaded();
        
        recognizerRef.current = recognizer;
        transferRecognizerRef.current = recognizer.createTransfer('bjarne-voice-1');
        
        setIsModelLoading(false);
      } catch (error) {
        console.error('Failed to load model:', error);
        alert('Kunde inte ladda ljudmodellen.');
      }
    };

    loadModel();

    return () => {
      isCollectingRef.current = false;
      if (transferRecognizerRef.current) {
        transferRecognizerRef.current.stopListening().catch(() => {});
      }
      // Note: samples in closure is stale ([]), so manual cleanup of ObjectURLs 
      // is handled via deleteSample or page reload.
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // --- Helper: Access Internal Dataset to sync UIDs ---
  const getLastExampleUid = (classId: string): string | null => {
      if (!transferRecognizerRef.current) return null;
      // @ts-ignore - accessing internal dataset to manage individual samples
      const dataset = transferRecognizerRef.current.dataset; 
      if (dataset) {
          const examples = dataset.getExamples(classId);
          if (examples && examples.length > 0) {
              return examples[examples.length - 1].uid;
          }
      }
      return null;
  };

  const removeExampleFromModel = (uid: string) => {
      if (!transferRecognizerRef.current) return;
      try {
          // @ts-ignore - accessing internal dataset
          transferRecognizerRef.current.dataset.removeExample(uid);
          console.log(`Removed example ${uid} from model`);
      } catch (e) {
          console.error("Failed to remove example from model:", e);
      }
  };

  // --- Data Collection ---
  const captureSample = async (classId: string, sessionId: number) => {
      if (!transferRecognizerRef.current) return;
      
      // Critical Lock: If already recording, do NOT start another one.
      if (isRecordingRef.current) return;
      isRecordingRef.current = true;

      try {
          // 1. Start collecting for Model (creates spectrogram)
          // collectExample listens for ~1 sec by default
          const collectPromise = transferRecognizerRef.current.collectExample(classId);

          // 2. Start collecting for UI (creates Blob)
          // We need a fresh stream for MediaRecorder to ensure we capture what the model hears
          const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
          const mediaRecorder = new MediaRecorder(stream);
          mediaRecorderRef.current = mediaRecorder;
          audioChunksRef.current = [];

          mediaRecorder.ondataavailable = (event) => {
              if (event.data.size > 0) {
                  audioChunksRef.current.push(event.data);
              }
          };

          mediaRecorder.onstop = () => {
              const blob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
              const url = URL.createObjectURL(blob);
              
              // Get the UID assigned by TFJS
              const tfUid = getLastExampleUid(classId);
              
              if (tfUid) {
                  const newSample: AudioSample = {
                      id: simpleUUID(),
                      tfUid,
                      classId,
                      blob,
                      url,
                      timestamp: Date.now()
                  };
                  
                  setSamples(prev => [...prev, newSample]);
              }
              
              // Stop tracks to release mic
              stream.getTracks().forEach(track => track.stop());
          };

          mediaRecorder.start();

          // Wait for model collection to finish (approx 1 sec)
          await collectPromise;
          
          // Stop recorder
          if (mediaRecorder.state !== 'inactive') {
              mediaRecorder.stop();
          }

      } catch (err) {
          console.error("Error collecting example:", err);
      } finally {
          // Release lock
          isRecordingRef.current = false;

          // Recursion Logic:
          // Only repeat if:
          // 1. User is still holding the button (isCollectingRef)
          // 2. The session ID matches the current session (User hasn't released and pressed again)
          
          if (isCollectingRef.current && currentSessionId.current === sessionId) {
              // Small delay to prevent CPU spam, but keep it snappy
              setTimeout(() => captureSample(classId, sessionId), 100); 
          }
      }
  };

  const startCollecting = useCallback(async (classId: string) => {
    // Use Pointer Events to prevent double firing on mobile/hybrid
    if (!transferRecognizerRef.current) return;
    if (isListening) await stopListening();
    
    // Mark intent to collect
    isCollectingRef.current = true;
    
    // Start a new session ID
    const newSessionId = Date.now();
    currentSessionId.current = newSessionId;

    // Attempt to capture. If locked, captureSample will simply return.
    // If locked, the previous loop will pick this up via isCollectingRef check at the end.
    captureSample(classId, newSessionId);
  }, [isListening]);

  const stopCollecting = useCallback(() => {
    isCollectingRef.current = false;
    // Note: We do NOT reset currentSessionId here. 
    // Leaving it "old" ensures any pending recursive calls from the previous session 
    // will fail their check (currentSessionId.current === sessionId) and stop.
  }, []);

  const deleteSample = (sample: AudioSample) => {
      // 1. Remove from UI
      setSamples(prev => prev.filter(s => s.id !== sample.id));
      // 2. Remove from Model
      removeExampleFromModel(sample.tfUid);
      // 3. Cleanup
      URL.revokeObjectURL(sample.url);
  };

  // --- Inference ---
  const startListening = useCallback(async () => {
    if (!transferRecognizerRef.current || isListening) return;
    
    if (!isTrainedRef.current) {
      alert("Du måste träna modellen först!");
      return;
    }

    // SAFETY CHECK: Ensure model has loaded words correctly
    const currentWords = transferRecognizerRef.current.wordLabels();
    if (!currentWords || currentWords.length === 0) {
        console.error("Kritisk: Modellens ordlista är fortfarande tom.");
        alert("Ett internt fel uppstod med modellen. Ordlistan saknas. Ladda om sidan och försök igen.");
        return;
    }

    setIsListening(true);
    try {
      await transferRecognizerRef.current.listen((result: speechCommands.SpeechCommandRecognizerResult) => {
        const scores = Array.from(result.scores as Float32Array);
        
        setProbabilities(scores);

        const maxScore = Math.max(...scores);
        const maxIndex = scores.indexOf(maxScore);
        
        // Get the word label from the recognizer's word list
        const words = transferRecognizerRef.current?.wordLabels();
        if (words) {
            const detectedWord = words[maxIndex];
            // Check if it maps to our YES class ID ('1')
            if (detectedWord === '1' && maxScore > THRESHOLD) {
                setDetectedLabel(LABELS.YES);
            } else {
                setDetectedLabel(null);
            }
        }

      }, {
        overlapFactor: 0.50, 
        includeSpectrogram: false,
        probabilityThreshold: 0.75,
        invokeCallbackOnNoiseAndUnknown: false 
      });
    } catch (err) {
      console.error("Listening error:", err);
      setIsListening(false);
    }
  }, [isListening]);

  const stopListening = async () => {
    if (!transferRecognizerRef.current || !isListening) return;
    await transferRecognizerRef.current.stopListening();
    setIsListening(false);
    setDetectedLabel(null);
    setProbabilities([0, 0]);
  };

  // --- Training ---
  const trainModel = async () => {
    if (!transferRecognizerRef.current) return;
    
    const counts = transferRecognizerRef.current.countExamples();
    const activeClasses = Object.keys(counts).filter(key => (counts[key] as number) > 0);
    
    if (activeClasses.length < 2) {
      alert("Du måste spela in både 'Bakgrundsbrus' och 'Ja' för att AI:n ska förstå skillnaden.");
      return;
    }

    setIsTraining(true);
    setTrainingProgress(null);
    if (isListening) await stopListening();

    try {
      await transferRecognizerRef.current.train({
        epochs: 25,
        callback: {
          onEpochEnd: (epoch, logs) => {
            setTrainingProgress({
              loss: logs?.loss || 0,
              accuracy: logs?.acc || 0
            });
          }
        }
      });
      
      isTrainedRef.current = true;
      setIsTrained(true);
      startListening();
    } catch (err) {
      console.error("Training error:", err);
      alert("Ett fel uppstod vid träningen.");
    } finally {
      setIsTraining(false);
    }
  };

  // --- ZIP Persistence (Robust) ---

  const saveProjectAsZip = async () => {
      if (!transferRecognizerRef.current || !isTrainedRef.current) return;

      try {
          setIsModelLoading(true);
          const zip = new JSZip();

          // 1. Save Model & Weights
          let modelArtifacts: any = null;
          await transferRecognizerRef.current.save({
              save: async (artifacts) => {
                  modelArtifacts = artifacts;
                  return {
                      modelArtifactsInfo: {
                          dateSaved: new Date(),
                          modelTopologyType: 'JSON',
                          modelTopologyBytes: 0,
                          weightSpecsBytes: 0,
                          weightDataBytes: 0,
                      }
                  };
              }
          });

          if (modelArtifacts) {
              const modelTopology = {
                  modelTopology: modelArtifacts.modelTopology,
                  weightsManifest: [{
                      paths: ["./model.weights.bin"],
                      weights: modelArtifacts.weightSpecs
                  }],
                  format: 'layers-model',
                  generatedBy: 'TensorFlow.js tfjs-layers v4.22.0',
                  convertedBy: null
              };
              zip.file("model.json", JSON.stringify(modelTopology, null, 2));

              if (modelArtifacts.weightData) {
                  zip.file("model.weights.bin", modelArtifacts.weightData);
              }
          }

          // 2. Save Metadata
          const metadata = {
              name: "Bjarne Project",
              created: new Date().toISOString(),
              wordLabels: transferRecognizerRef.current.wordLabels(),
              version: "2.0 (ZIP)"
          };
          zip.file("metadata.json", JSON.stringify(metadata, null, 2));

          // 3. Save Audio Samples
          const dataFolder = zip.folder("data");
          samples.forEach((sample) => {
              const classInfo = INITIAL_CLASSES.find(c => c.id === sample.classId);
              const folderName = classInfo ? `${classInfo.id}_${classInfo.name.replace(/[^a-z0-9]/gi, '_')}` : `unknown_${sample.classId}`;
              const classFolder = dataFolder?.folder(folderName);
              classFolder?.file(`${sample.id}.webm`, sample.blob);
          });

          // 4. Generate
          const content = await zip.generateAsync({ type: "blob" });
          const url = URL.createObjectURL(content);
          const a = document.createElement('a');
          a.href = url;
          a.download = `bjarne_projekt_${new Date().toISOString().slice(0,10)}.zip`;
          document.body.appendChild(a);
          a.click();
          document.body.removeChild(a);
          URL.revokeObjectURL(url);

      } catch (err) {
          console.error("Save Zip Failed:", err);
          alert("Kunde inte skapa ZIP-filen.");
      } finally {
          setIsModelLoading(false);
      }
  };

  const loadProjectFromZip = async (event: React.ChangeEvent<HTMLInputElement>) => {
      const file = event.target.files?.[0];
      if (!file) return;
      
      if (isListening) await stopListening();
      setIsModelLoading(true);

      try {
          // Robustly handle JSZip instantiation if import is a module namespace
          let zipInstance: JSZip;
          if (typeof JSZip === 'function') {
             zipInstance = await JSZip.loadAsync(file);
          } else {
             // @ts-ignore handles synthetic default import mismatch on some CDNs
             zipInstance = await JSZip.loadAsync(file); 
          }
          const zip = zipInstance;
          console.log("ZIP loaded. Analyzing contents...");

          // --- SEARCH FOR MODEL FILES RECURSIVELY ---
          // 1. Find model.json
          let modelEntry: JSZip.JSZipObject | null = null;
          let modelPath = "";
          
          zip.forEach((relativePath, zipEntry) => {
              if (relativePath.endsWith("model.json") && !zipEntry.dir) {
                  modelEntry = zipEntry;
                  modelPath = relativePath;
              }
          });

          let modelLoaded = false;
          let rootPrefix = ""; // e.g. "MyProject/"
          let failureReason = "";

          if (modelEntry) {
              console.log("Found model.json at:", modelPath);
              const lastSlash = modelPath.lastIndexOf('/');
              rootPrefix = lastSlash !== -1 ? modelPath.substring(0, lastSlash + 1) : "";

              try {
                  // Read model.json to interpret manifest
                  // @ts-ignore
                  const modelJsonText = await modelEntry.async("string");
                  const modelJSON = JSON.parse(modelJsonText);
                  
                  let weightData: ArrayBuffer | null = null;

                  // A. Try to find weights via Manifest (Best practice)
                  if (modelJSON.weightsManifest && modelJSON.weightsManifest.length > 0) {
                      const paths = modelJSON.weightsManifest[0].paths; // e.g. ["./model.weights.bin"]
                      if (paths && paths.length > 0) {
                          // Clean filename (remove ./ prefix)
                          const manifestWeightName = paths[0].replace(/^\.\//, '');
                          const fullWeightPath = rootPrefix + manifestWeightName;
                          
                          console.log("Manifest expects weights at:", fullWeightPath);
                          const weightEntry = zip.file(fullWeightPath);
                          if (weightEntry) {
                              weightData = await weightEntry.async("arraybuffer");
                          }
                      }
                  }

                  // B. Fallback: Try common names if manifest failed or wasn't clear
                  if (!weightData) {
                      console.log("Weights not found via manifest path. Trying common names...");
                      const fallbackNames = ["model.weights.bin", "weights.bin", "group1-shard1of1.bin"];
                      for (const name of fallbackNames) {
                          const f = zip.file(rootPrefix + name);
                          if (f) {
                              console.log("Found weights fallback:", name);
                              weightData = await f.async("arraybuffer");
                              break;
                          }
                      }
                  }

                  if (weightData) {
                       // Setup Loader
                       const customLoader = {
                          load: async () => {
                              let weightSpecs: tf.io.WeightsManifestEntry[] = [];
                              if (modelJSON.weightsManifest) {
                                  // @ts-ignore
                                  modelJSON.weightsManifest.forEach((g: any) => weightSpecs.push(...g.weights));
                              }
                              return {
                                  modelTopology: modelJSON.modelTopology,
                                  weightSpecs: weightSpecs,
                                  weightData: weightData as ArrayBuffer,
                                  format: 'layers-model',
                                  generatedBy: modelJSON.generatedBy,
                                  convertedBy: modelJSON.convertedBy
                              };
                          }
                      };

                      // Find Metadata
                      // 1. Try standard location
                      let metadataText = await zip.file(rootPrefix + "metadata.json")?.async("string");
                      
                      // 2. Scan for ANY json with 'wordLabels' if standard not found
                      if (!metadataText) {
                          console.log("metadata.json not found at root. Scanning other JSONs...");
                          const jsonFiles = zip.filter((path, file) => path.endsWith('.json') && path !== modelPath);
                          for (const jf of jsonFiles) {
                              const txt = await jf.async("string");
                              if (txt.includes('wordLabels')) {
                                  metadataText = txt;
                                  console.log("Found alternative metadata file:", jf.name);
                                  break;
                              }
                          }
                      }

                      let metadataObj = metadataText ? JSON.parse(metadataText) : {};
                      
                      // Verify we have labels before loading
                      if (!metadataObj.wordLabels) {
                          console.warn("Warning: No wordLabels found in metadata. Model might not work for classification.");
                      }

                      // Load into SpeechCommands
                      // @ts-ignore
                      await transferRecognizerRef.current?.load(customLoader, metadataObj);

                      if (metadataObj.wordLabels) {
                          (transferRecognizerRef.current as any).words = metadataObj.wordLabels;
                      }
                      
                      modelLoaded = true;
                      console.log("Model loaded successfully.");
                  } else {
                      failureReason = "Kunde inte hitta vikt-filen (.bin) som hör till modellen.";
                  }
              } catch (e) {
                  console.warn("Failed to parse/load model files:", e);
                  failureReason = "Modellfilerna verkar skadade eller inkompatibla.";
              }
          } else {
              failureReason = "Hittade ingen 'model.json' i ZIP-filen.";
          }

          // --- SEARCH FOR AUDIO FILES RECURSIVELY ---
          
          const newSamples: AudioSample[] = [];
          const filePromises: Promise<void>[] = [];
          setSamples([]); 

          zip.forEach((relativePath, zipEntry) => {
              if (zipEntry.dir) return;
              if (!relativePath.match(/\.(webm|wav|ogg)$/i)) return; // Only audio files

              // Attempt to detect Class ID from folder structure
              const parts = relativePath.split('/');
              
              let matchedClassId: string | null = null;
              
              // Strategy 1: Check Parent Folder Name
              if (parts.length >= 2) {
                  const folderName = parts[parts.length - 2];
                  if (folderName.startsWith("0")) matchedClassId = "0";
                  else if (folderName.startsWith("1")) matchedClassId = "1";
                  else {
                      const found = INITIAL_CLASSES.find(c => folderName.toLowerCase().includes(c.name.toLowerCase()));
                      if (found) matchedClassId = found.id;
                  }
              }

              // Strategy 2: If still null, check Filename keywords (Robustness for flat zips)
              if (!matchedClassId) {
                  const fileName = parts[parts.length - 1].toLowerCase();
                  if (fileName.includes("bakgrund") || fileName.includes("brus") || fileName.includes("noise")) matchedClassId = "0";
                  else if (fileName.includes("hm") || fileName.includes("ja") || fileName.includes("yes")) matchedClassId = "1";
              }

              if (matchedClassId) {
                  const p = (async () => {
                      try {
                          const blob = await zipEntry.async("blob");
                          const url = URL.createObjectURL(blob);
                          newSamples.push({
                              id: simpleUUID(),
                              tfUid: 'imported-' + simpleUUID(),
                              classId: matchedClassId!,
                              blob: blob,
                              url: url,
                              timestamp: Date.now()
                          });
                      } catch (err) {
                          console.warn("Failed to read audio:", relativePath);
                      }
                  })();
                  filePromises.push(p);
              }
          });

          await Promise.all(filePromises);
          setSamples(prev => [...prev, ...newSamples]);

          if (modelLoaded) {
              setIsTrained(true);
              isTrainedRef.current = true;
              alert(`Projekt laddat!\n\nModell: OK\nLjudfiler: ${newSamples.length} st`);
              startListening();
          } else {
              setIsTrained(false);
              isTrainedRef.current = false;
              // More descriptive error
              alert(`ZIP inläst, men kunde inte aktivera AI:n.\n\nOrsak: ${failureReason}\n\nLjudfiler hittade: ${newSamples.length} st.\n(Om du har ljudfiler kan du träna en ny modell nu).`);
          }

      } catch (err) {
          console.error("Load ZIP Failed:", err);
          alert("Kunde inte läsa ZIP-filen. Är den skadad?");
      } finally {
          setIsModelLoading(false);
          if (zipInputRef.current) zipInputRef.current.value = '';
      }
  };

  const triggerZipLoad = () => zipInputRef.current?.click();

  return (
    <div className="space-y-8">
      {/* Hidden File Inputs */}
      <input 
        type="file" 
        ref={zipInputRef}
        onChange={loadProjectFromZip}
        accept=".zip"
        style={{ display: 'none' }}
      />

      {/* Status Header */}
      <div className="bg-slate-800 rounded-xl p-6 border border-slate-700 shadow-lg">
        <div className="flex flex-col lg:flex-row items-center justify-between gap-4">
          <div className="flex items-center space-x-4 w-full lg:w-auto">
            <div className={`h-3 w-3 rounded-full animate-pulse-fast ${isModelLoading ? 'bg-yellow-500' : 'bg-green-500'}`} />
            <div>
              <h2 className="font-semibold text-lg">Systemstatus</h2>
              <p className="text-slate-400 text-sm">
                {isModelLoading ? 'Arbetar...' : isTrained ? 'Redo för igenkänning' : 'Väntar på träning'}
              </p>
            </div>
          </div>
          
          {/* Save/Load Controls */}
          <div className="flex flex-col items-end gap-2 w-full lg:w-auto">
              <div className="flex flex-wrap gap-2 w-full justify-end">
                
                {/* New ZIP Load Button */}
                <button 
                  onClick={triggerZipLoad}
                  className="px-4 py-2 bg-slate-700 hover:bg-slate-600 rounded-lg text-sm font-medium transition-colors flex items-center gap-2"
                  disabled={isModelLoading}
                >
                  📂 Öppna Projekt (.zip)
                </button>

                {/* Save Button */}
                <button 
                  onClick={saveProjectAsZip}
                  disabled={!isTrained}
                  className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors flex items-center gap-2 ${
                      !isTrained ? 'bg-slate-700/50 text-slate-500 cursor-not-allowed' : 'bg-blue-600 hover:bg-blue-500 text-white'
                  }`}
                >
                  💾 Spara Projekt (.zip)
                </button>
             </div>
             <p className="text-xs text-slate-500 hidden sm:block">
                Sparar modell och alla ljudfiler i ett paket.
             </p>
          </div>
        </div>
      </div>

      {/* Main Training Interface */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {INITIAL_CLASSES.map((cls) => (
          <div 
            key={cls.id}
            className="relative overflow-hidden rounded-2xl border border-slate-700 bg-slate-800/50 p-6 transition-all hover:border-slate-600"
          >
            <div className="flex items-center justify-between mb-6">
              <div className="flex items-center gap-3">
                <span className="text-3xl">{cls.icon}</span>
                <h3 className="text-xl font-bold">{cls.name}</h3>
              </div>
              <div className="px-3 py-1 rounded-full bg-slate-700 text-xs font-mono">
                {samples.filter(s => s.classId === cls.id).length} samples
              </div>
            </div>

            <button
              onPointerDown={(e) => {
                  e.preventDefault(); // Important to prevent mouse emulation
                  startCollecting(cls.id);
              }}
              onPointerUp={stopCollecting}
              onPointerLeave={stopCollecting}
              disabled={isTraining || isModelLoading}
              className={`w-full py-12 rounded-xl text-center transition-all transform active:scale-95 touch-none select-none ${
                isTraining ? 'opacity-50 cursor-wait' : 
                'hover:brightness-110 active:brightness-125 cursor-pointer shadow-lg'
              } ${cls.color}`}
            >
              <div className="text-2xl font-bold mb-2">Håll för att spela in</div>
              <div className="text-white/80 text-sm">Släpp för att spara</div>
            </button>

            {/* Sample List (Scrollable) */}
            <div className="mt-4 h-32 overflow-y-auto space-y-2 pr-2 scrollbar-thin scrollbar-thumb-slate-600 scrollbar-track-transparent">
                {samples.filter(s => s.classId === cls.id).map((sample) => (
                    <div key={sample.id} className="flex items-center justify-between bg-slate-900/50 p-2 rounded text-xs group">
                        <span className="text-slate-400 font-mono">{new Date(sample.timestamp).toLocaleTimeString()}</span>
                        <div className="flex gap-2">
                            <button 
                                onClick={() => { const a = new Audio(sample.url); a.play(); }}
                                className="text-blue-400 hover:text-blue-300 px-2"
                            >
                                ▶️
                            </button>
                            <button 
                                onClick={() => deleteSample(sample)}
                                className="text-red-400 hover:text-red-300 px-2 opacity-0 group-hover:opacity-100 transition-opacity"
                            >
                                🗑️
                            </button>
                        </div>
                    </div>
                ))}
            </div>
          </div>
        ))}
      </div>

      {/* Training Controls */}
      <div className="flex flex-col items-center justify-center py-6">
        <button
          onClick={trainModel}
          disabled={isTraining || samples.length < 2}
          className={`relative px-8 py-4 rounded-full font-bold text-lg transition-all transform hover:scale-105 shadow-xl ${
            isTraining ? 'bg-slate-700 cursor-wait' :
            samples.length < 2 ? 'bg-slate-700 opacity-50 cursor-not-allowed' :
            'bg-gradient-to-r from-indigo-500 to-purple-600 hover:from-indigo-400 hover:to-purple-500'
          }`}
        >
          {isTraining ? (
            <span className="flex items-center gap-2">
              <span className="animate-spin">⚙️</span> Tränar ({trainingProgress ? Math.round(trainingProgress.accuracy * 100) : 0}%)...
            </span>
          ) : '🚀 Starta Träning'}
        </button>
        
        {trainingProgress && !isTraining && (
          <div className="mt-4 text-emerald-400 font-mono text-sm">
            Träning klar! Noggrannhet: {Math.round(trainingProgress.accuracy * 100)}%
          </div>
        )}
      </div>

      {/* Live Inference Feedback */}
      {isTrained && (
        <div className="border-t border-slate-700 pt-8">
          <h3 className="text-center text-slate-400 mb-6 uppercase tracking-widest text-sm">Realtidsanalys</h3>
          
          <div className="flex justify-center gap-4 mb-8">
            {INITIAL_CLASSES.map((cls, idx) => (
              <div key={cls.id} className="flex flex-col items-center w-24">
                 <div className="h-32 w-8 bg-slate-800 rounded-full relative overflow-hidden">
                    <div 
                        className={`absolute bottom-0 w-full transition-all duration-100 ${cls.color}`}
                        style={{ height: `${probabilities[idx] * 100}%` }}
                    />
                 </div>
                 <span className="mt-2 text-xs font-mono text-slate-400">{Math.round(probabilities[idx] * 100)}%</span>
              </div>
            ))}
          </div>

          {/* BIG RESULT INDICATOR */}
          <div className={`max-w-md mx-auto p-8 rounded-3xl text-center transition-all duration-200 transform ${
             detectedLabel ? 'bg-emerald-500 scale-110 shadow-[0_0_50px_rgba(16,185,129,0.4)]' : 'bg-slate-800'
          }`}>
             <div className="text-6xl mb-4">
                {detectedLabel ? '👍' : '👂'}
             </div>
             <h2 className="text-3xl font-bold text-white">
                {detectedLabel || 'Lyssnar...'}
             </h2>
          </div>
        </div>
      )}
    </div>
  );
};

export default BjarneVoiceTrainer;