import { useState, useEffect, useRef } from 'react';

const CONFIG = {
  CONFIDENCE_THRESHOLD: 0.85, 
  WARNING_RATIO: 0.6,         
  MOUTH_OPEN_THRESHOLD: 0.02, 
  INFERENCE_INTERVAL: 250
};

function App() {
  const [status, setStatus] = useState('Initializing...');
  const [trafficLight, setTrafficLight] = useState('OFF');
  const [debugInfo, setDebugInfo] = useState({ label: '-', score: 0, mouth: 'Closed' });
  
  const videoRef = useRef(null);
  const classifierRef = useRef(null);
  const landmarkerRef = useRef(null);
  const historyQueue = useRef([]);
  const isRunning = useRef(false);
  const isInitCalled = useRef(false);

  // 👇 모델이 원하는 설정값을 저장할 변수
  const modelSettings = useRef({ frequency: 16000, inputSize: 16000 });

  useEffect(() => {
    if (isInitCalled.current) return;
    isInitCalled.current = true;

    const loadModels = async () => {
      try {
        console.log("🚀 System Start: Loading...");

        // 1. Audio Model Loading
        console.log("1️⃣ Waiting for EdgeImpulseClassifier...");
        let retries = 0;
        while (!window.EdgeImpulseClassifier && retries < 100) {
            await new Promise(r => setTimeout(r, 100));
            retries++;
        }
        if (!window.EdgeImpulseClassifier) throw new Error("Audio Model Timeout");

        const classifier = new window.EdgeImpulseClassifier();
        await classifier.init();
        
        // 🔍 [중요] 모델한테 직접 물어봅니다: "데이터 몇 개 줘야 해?"
        const props = classifier.getProperties();
        console.log("📏 Model Properties:", props);
        modelSettings.current = {
            frequency: props.frequency,             // 예: 16000
            inputSize: props.input_features_count   // 예: 16000 (또는 다른 값)
        };
        console.log(`✅ Audio Model Configured: Needs ${modelSettings.current.inputSize} samples at ${modelSettings.current.frequency}Hz`);

        classifierRef.current = classifier;

        // 1.5 Backup Audio Module
        console.log("🔒 Backing up Audio Module...");
        const audioModuleBackup = window.Module;
        window.Module = undefined;

        // 2. Vision Model Loading
        console.log("2️⃣ Loading Vision Model...");
        const { FilesetResolver, FaceLandmarker } = await import('@mediapipe/tasks-vision');
        const vision = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm");
        landmarkerRef.current = await FaceLandmarker.createFromOptions(vision, {
          baseOptions: { modelAssetPath: `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task` },
          runningMode: "VIDEO",
          numFaces: 1
        });
        console.log("✅ Vision Model Ready!");

        // 🔄 Restore
        console.log("🔓 Restoring Audio Module...");
        window.Module = audioModuleBackup;

        setStatus('Ready to Start! 🚀');

      } catch (error) {
        console.error("❌ Error:", error);
        setStatus('Error: Check Console');
      }
    };

    loadModels();
  }, []);

  // 다운샘플링 함수
  const downsampleBuffer = (buffer, inputRate, outputRate) => {
    if (outputRate === inputRate) return buffer;
    const sampleRateRatio = inputRate / outputRate;
    const newLength = Math.round(buffer.length / sampleRateRatio);
    const result = new Float32Array(newLength);
    let offsetResult = 0;
    let offsetBuffer = 0;
    
    while (offsetResult < result.length) {
        const nextOffsetBuffer = Math.round((offsetResult + 1) * sampleRateRatio);
        let accum = 0, count = 0;
        for (let i = offsetBuffer; i < nextOffsetBuffer && i < buffer.length; i++) {
            accum += buffer[i];
            count++;
        }
        result[offsetResult] = accum / count;
        offsetResult++;
        offsetBuffer = nextOffsetBuffer;
    }
    return result;
  };

  const startSystem = async () => {
    try {
      setStatus('Starting Sensors...');
      
      const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
      videoRef.current.srcObject = stream;
      videoRef.current.play();

      const AudioContext = window.AudioContext || window.webkitAudioContext;
      const audioCtx = new AudioContext();
      const actualSampleRate = audioCtx.sampleRate;
      const targetFreq = modelSettings.current.frequency; // 모델이 원하는 주파수 사용

      console.log(`🎤 Mic: ${actualSampleRate}Hz -> Model: ${targetFreq}Hz`);

      if (audioCtx.state === 'suspended') await audioCtx.resume();

      const source = audioCtx.createMediaStreamSource(stream);
      const processor = audioCtx.createScriptProcessor(16384, 1, 1);
      source.connect(processor);
      processor.connect(audioCtx.destination); 

      const audioBuffer = [];
      
      processor.onaudioprocess = (e) => {
        const input = e.inputBuffer.getChannelData(0);
        // 모델이 원하는 주파수로 변환
        const downsampled = downsampleBuffer(input, actualSampleRate, targetFreq);
        
        for (let i = 0; i < downsampled.length; i++) audioBuffer.push(downsampled[i]);
        
        // 버퍼가 너무 커지지 않게 관리 (필요한 크기의 5배 유지)
        const maxBufferSize = modelSettings.current.inputSize * 5;
        if (audioBuffer.length > maxBufferSize) {
            audioBuffer.splice(0, audioBuffer.length - (modelSettings.current.inputSize * 2));
        }
      };

      isRunning.current = true;
      setStatus('Monitoring Active 🟢');

      setInterval(async () => {
        if (!isRunning.current) return;

        // Vision
        let isMouthOpen = false;
        if (landmarkerRef.current && videoRef.current.currentTime > 0) {
          const result = landmarkerRef.current.detectForVideo(videoRef.current, performance.now());
          if (result.faceLandmarks.length > 0) {
            const upperLip = result.faceLandmarks[0][13].y;
            const lowerLip = result.faceLandmarks[0][14].y;
            if ((lowerLip - upperLip) > CONFIG.MOUTH_OPEN_THRESHOLD) isMouthOpen = true;
          }
        }

        // Audio
        let audioLabel = 'noise';
        let audioConfidence = 0;
        
        const requiredSize = modelSettings.current.inputSize; // 모델이 원하는 데이터 개수

        // 데이터가 충분히 모였는지 확인
        if (classifierRef.current && audioBuffer.length >= requiredSize) {
           // 정확히 필요한 만큼만 잘라서 가져옴 (Slice)
           const inputData = audioBuffer.slice(audioBuffer.length - requiredSize);
           
           try {
              const res = classifierRef.current.classify(inputData);
              if (res.results && res.results.length > 0) {
                  const topResult = res.results.reduce((prev, current) => (prev.value > current.value) ? prev : current);
                  audioLabel = topResult.label;
                  audioConfidence = topResult.value;
                  
                  console.log(`🗣️ ${topResult.label}: ${(topResult.value * 100).toFixed(0)}%`);
              }
           } catch(e) { 
              // console.warn("Skipping frame:", e);
           }
        }

        // Logic
        let isKoreanSuspected = 0;
        if (isMouthOpen && audioLabel === 'korean' && audioConfidence > CONFIG.CONFIDENCE_THRESHOLD) { 
            isKoreanSuspected = 1;
        }

        historyQueue.current.push(isKoreanSuspected);
        if (historyQueue.current.length > 10) historyQueue.current.shift();
        
        const suspectCount = historyQueue.current.filter(v => v === 1).length;
        if (suspectCount >= 3) setTrafficLight('RED');
        else if (suspectCount >= 1) setTrafficLight('YELLOW');
        else setTrafficLight('GREEN');

        setDebugInfo({ label: audioLabel, score: audioConfidence, mouth: isMouthOpen ? 'OPEN 😲' : 'Closed 😐' });
      }, CONFIG.INFERENCE_INTERVAL);

    } catch (err) {
      console.error(err);
      setStatus('Error: Check Permissions');
    }
  };

  return (
    <div className="min-h-screen bg-gray-900 text-white flex flex-col items-center justify-center p-4">
      <h1 className="text-3xl font-bold mb-6 text-blue-400">OnlyEnglish Pro 🇺🇸</h1>
      <div className={`w-64 h-64 rounded-full border-8 border-gray-700 flex items-center justify-center mb-8 transition-all duration-300
        ${trafficLight === 'RED' ? 'bg-red-600 shadow-[0_0_60px_red] animate-pulse' : 
          trafficLight === 'YELLOW' ? 'bg-yellow-500 shadow-[0_0_40px_yellow]' : 
          trafficLight === 'GREEN' ? 'bg-green-600 shadow-[0_0_40px_green]' : 'bg-gray-800'}`}>
        <span className="text-5xl font-bold">{trafficLight}</span>
      </div>
      <p className="text-xl mb-6 text-gray-300 font-mono animate-pulse">{status}</p>
      {!isRunning.current && <button onClick={startSystem} className="px-8 py-3 bg-blue-600 rounded-full font-bold">👉 Start Class</button>}
      <div className="mt-8 bg-gray-800 p-4 rounded-xl w-full max-w-sm border border-gray-700">
        <div className="grid grid-cols-2 gap-2 text-sm">
          <span>Sound:</span> <span className="font-bold text-yellow-300">{debugInfo.label.toUpperCase()}</span>
          <span>Score:</span> <span className="font-mono">{Math.round(debugInfo.score * 100)}%</span>
          <span>Mouth:</span> <span className={debugInfo.mouth.includes('OPEN') ? 'text-red-400 font-bold' : 'text-green-400'}>{debugInfo.mouth}</span>
        </div>
      </div>
      <video ref={videoRef} className="opacity-0 fixed top-0 left-0 w-1 h-1" autoPlay playsInline muted></video>
    </div>
  );
}

export default App;
