import { useState, useEffect, useRef } from 'react';
import './index.css'; 

function App() {
  const [isRunning, setIsRunning] = useState(false);
  const [statusText, setStatusText] = useState("System Standby");
  const [showSettings, setShowSettings] = useState(false);
  const [showLogs, setShowLogs] = useState(true); 
  const [settings, setSettings] = useState({ confidence: 50, mouthOpen: 4, lipMovement: 20, strictness: 3 });
  const [logs, setLogs] = useState([]);
  const [detectedStudents, setDetectedStudents] = useState({});
  const [audioStatus, setAudioStatus] = useState({ label: 'Standby', score: 0 });
  const [videoStatus, setVideoStatus] = useState({ state: 'Closed', gap: 0 });

  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const classifierRef = useRef(null); 
  const audioCtxRef = useRef(null);   
  const lipHistory = useRef([]);
  const violationQueue = useRef([]);
  const alertTimeout = useRef(null);
  const dbRef = useRef(null);

  // 초기화 (Firebase 연결)
  useEffect(() => {
    if (window.firebase && !dbRef.current) {
      const firebaseConfig = {
        apiKey: "AIzaSyDaLlrTKsMCpCzVgBW9icTmEPcuO_zoWVY",
        authDomain: "acdt-project.firebaseapp.com",
        projectId: "acdt-project",
        storageBucket: "acdt-project.firebasestorage.app",
        messagingSenderId: "243281762920",
        appId: "1:243281762920:web:6641d9eadfe1e93442f9dd",
        measurementId: "G-J9TVZXN3LE"
      };
      if (!window.firebase.apps.length) { window.firebase.initializeApp(firebaseConfig); }
      dbRef.current = window.firebase.firestore();
      loadList();
    }
  }, []);

  const calculateStandardDeviation = (arr) => {
    if (arr.length === 0) return 0;
    const mean = arr.reduce((a, b) => a + b, 0) / arr.length;
    const variance = arr.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / arr.length;
    return Math.sqrt(variance);
  };

  const downsampleBuffer = (buffer, sampleRate, outSampleRate) => {
    if (outSampleRate === sampleRate) return buffer;
    let sampleRateRatio = sampleRate / outSampleRate;
    let newLength = Math.round(buffer.length / sampleRateRatio);
    let result = new Float32Array(newLength);
    let offsetResult = 0; let offsetBuffer = 0;
    while (offsetResult < result.length) {
      let nextOffsetBuffer = Math.round((offsetResult + 1) * sampleRateRatio);
      let accum = 0, count = 0;
      for (let i = offsetBuffer; i < nextOffsetBuffer && i < buffer.length; i++) { accum += buffer[i]; count++; }
      result[offsetResult] = accum / count; offsetResult++; offsetBuffer = nextOffsetBuffer;
    }
    return result;
  };

  const startSystem = async () => {
    const name = document.getElementById('input-name').value;
    const id = document.getElementById('input-id').value;
    if (!name || !id) { alert("Please enter Name and Student ID!"); return; }

    setStatusText("Initializing AI...");
    try {
      const classifier = new window.EdgeImpulseClassifier();
      await classifier.init();
      classifierRef.current = classifier;
      await startAudioProcessing();
      await startFaceMesh();
      setIsRunning(true);
      setStatusText("Monitoring Active 🟢");
      document.getElementById('placeholder').style.display = 'none';
      document.getElementById('camera-wrapper').style.display = 'block';
      document.getElementById('status-panel').style.display = 'flex';
    } catch (err) {
      console.error(err);
      alert("Error: " + err.message);
      window.location.reload();
    }
  };

  const startAudioProcessing = async () => {
    const AudioContext = window.AudioContext || window.webkitAudioContext;
    const audioCtx = new AudioContext();
    audioCtxRef.current = audioCtx;
    if (audioCtx.state === 'suspended') await audioCtx.resume();
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const source = audioCtx.createMediaStreamSource(stream);
    const processor = audioCtx.createScriptProcessor(4096, 1, 1);
    const targetRate = 16000;
    const bufferSize = 16000; 
    let circularBuffer = new Float32Array(bufferSize);
    let writeIndex = 0;
    source.connect(processor);
    processor.connect(audioCtx.destination);
    processor.onaudioprocess = (e) => {
      if (!classifierRef.current) return;
      const inputData = e.inputBuffer.getChannelData(0);
      const downsampled = downsampleBuffer(inputData, audioCtx.sampleRate, targetRate);
      for (let i = 0; i < downsampled.length; i++) { circularBuffer[writeIndex] = downsampled[i]; writeIndex = (writeIndex + 1) % bufferSize; }
      let linearBuffer = new Float32Array(bufferSize);
      for (let i = 0; i < bufferSize; i++) { linearBuffer[i] = circularBuffer[(writeIndex + i) % bufferSize]; }
      try {
        let results = classifierRef.current.classify(linearBuffer);
        let top = results.results.reduce((p, c) => p.value > c.value ? p : c);
        setAudioStatus({ label: top.label, score: top.value });
      } catch (ex) {}
    };
  };

  const startFaceMesh = async () => {
    const videoElement = document.getElementById('input_video');
    const canvasElement = canvasRef.current;
    const ctx = canvasElement.getContext('2d');
    const faceMesh = new window.FaceMesh({ locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}` });
    faceMesh.setOptions({ maxNumFaces: 1, refineLandmarks: true, minDetectionConfidence: 0.5 });
    faceMesh.onResults((results) => {
      canvasElement.width = 500; canvasElement.height = 500;
      ctx.fillStyle = "black"; ctx.fillRect(0, 0, 500, 500);
      let currentVisualState = "Closed";
      let gapPercent = 0;
      if (results.image && results.multiFaceLandmarks.length > 0) {
        const lm = results.multiFaceLandmarks[0];
        const sW = videoElement.videoWidth; const sH = videoElement.videoHeight;
        const upper = lm[13]; const lower = lm[14];
        const zoom = 4.0; const cw = sW / zoom; const ch = sH / zoom;
        let cx = ((upper.x + lower.x) / 2 * sW) - cw / 2;
        let cy = ((upper.y + lower.y) / 2 * sH) - ch / 2;
        ctx.drawImage(results.image, cx, cy, cw, ch, 0, 0, 500, 500);
        const gap = lower.y - upper.y;
        lipHistory.current.push(gap);
        if (lipHistory.current.length > 5) lipHistory.current.shift();
        const movement = calculateStandardDeviation(lipHistory.current);
        gapPercent = (gap / 0.05) * 100;
        const mouthThreshold = settings.mouthOpen / 1000; 
        const moveThreshold = settings.lipMovement / 10000;
        if (gap > mouthThreshold) {
          if (movement > moveThreshold) currentVisualState = "Speaking";
          else currentVisualState = "Open";
        }
        checkViolation(currentVisualState);
      }
      setVideoStatus({ state: currentVisualState, gap: gapPercent });
    });
    const camera = new window.Camera(videoElement, { onFrame: async () => { await faceMesh.send({ image: videoElement }); }, width: 1280, height: 720 });
    await camera.start();
  };

  const checkViolation = (visualState) => {
    const isKorean = audioStatus.label === 'korean' && audioStatus.score > (settings.confidence / 100);
    const isMouthActive = visualState === "Speaking";
    if (isKorean && isMouthActive) { violationQueue.current.push(1); } else { violationQueue.current.push(0); }
    if (violationQueue.current.length > 10) violationQueue.current.shift();
    const violationCount = violationQueue.current.filter(v => v === 1).length;
    if (violationCount >= settings.strictness) { triggerDetection(); addLog("VIOLATION", 100, "Detected"); } 
    else if (Math.random() < 0.02) { addLog(audioStatus.label, Math.round(audioStatus.score * 100), visualState); }
  };

  const triggerDetection = () => {
    if (alertTimeout.current) return;
    const overlay = document.getElementById('alert-overlay');
    if(overlay) overlay.style.display = 'block';
    const img = document.getElementById('monitor-image');
    if(img) { img.src = "2.jpg"; img.classList.add('alert-mode'); }
    const name = document.getElementById('input-name').value;
    const id = document.getElementById('input-id').value;
    if (dbRef.current) { dbRef.current.collection("detections").add({ name: name, studentId: id, reason: "Korean Speaking", timestamp: window.firebase.firestore.FieldValue.serverTimestamp() }); }
    alertTimeout.current = setTimeout(() => {
      if(overlay) overlay.style.display = 'none';
      if(img) { img.src = "1.jpg"; img.classList.remove('alert-mode'); }
      alertTimeout.current = null;
      violationQueue.current = [];
    }, 3000);
  };

  const addLog = (label, score, visual) => {
    const time = new Date().toLocaleTimeString('en-US', { hour12: false });
    const newLog = { id: Date.now(), time, label, score, visual };
    setLogs(prev => [newLog, ...prev].slice(0, 20));
  };

  const loadList = () => {
    if (!dbRef.current) return;
    dbRef.current.collection("detections").orderBy("timestamp", "desc").onSnapshot(snapshot => {
        const students = {};
        snapshot.forEach(doc => {
          const data = doc.data();
          if (!students[data.studentId]) { students[data.studentId] = { name: data.name, id: data.studentId, records: [] }; }
          students[data.studentId].records.push({ time: data.timestamp ? new Date(data.timestamp.toDate()).toLocaleTimeString() : "Just now" });
        });
        setDetectedStudents(students);
      });
  };

  const manualAdd = () => {
    const n = document.getElementById('add-name').value;
    const i = document.getElementById('add-id').value;
    if(n && i && dbRef.current) { dbRef.current.collection("detections").add({name: n, studentId: i, timestamp: window.firebase.firestore.FieldValue.serverTimestamp()}); }
  };

  const deleteAllData = async () => {
    if(window.confirm("Delete ALL?")) { 
        const snap = await dbRef.current.collection("detections").get(); 
        const batch = dbRef.current.batch(); 
        snap.docs.forEach(d => batch.delete(d.ref)); 
        await batch.commit(); 
    }
  };

  return (
    <>
      <div id="sidebar">
        <div className="title-container"><h2>Korean Killer</h2><div id="kk-logo">KK</div></div>
        <input type="text" id="input-name" placeholder="Name" />
        <input type="text" id="input-id" placeholder="Student ID" />
        {!isRunning && <button id="btn-start" onClick={startSystem}>▶ Start Monitoring</button>}
        <div id="loading-msg" style={{ display: isRunning ? 'none' : 'none' }}>Initializing...</div>
        {isRunning && <button id="btn-stop" style={{display:'block'}} onClick={() => window.location.reload()}>⏹ Stop System</button>}
        <button id="btn-list" onClick={() => document.getElementById('list-panel').classList.toggle('open')}>📋 Detection List</button>
        <div id="prof-controls">
          <p>👮‍♂️ [Professor Mode]</p>
          <input type="text" id="add-name" placeholder="Name" style={{marginBottom:'5px'}} />
          <input type="text" id="add-id" placeholder="ID" style={{marginBottom:'5px'}} />
          <button onClick={manualAdd} style={{background:'#ff9800'}}>Manual Add</button>
          <hr style={{borderColor:'#555', margin:'15px 0'}} />
          <button onClick={deleteAllData} className="btn-delete-all">⚠️ DELETE ALL DATA</button>
        </div>
        <button id="btn-prof" onClick={() => { if (prompt("Password:") === "kyj") { document.getElementById('prof-controls').style.display = 'block'; document.getElementById('btn-prof').style.display = 'none'; } }}>🔒 Admin Auth</button>
      </div>

      <div id="list-panel">
        <h3>🚨 Detected Students</h3>
        <ul id="student-list" style={{listStyle:'none', padding:0}}>
          {Object.values(detectedStudents).map(student => (
            <li key={student.id} className={`student-item ${student.records.length > 3 ? 'problematic' : ''}`}>
              <div className="item-header">
                <div className="student-info"><b>{student.name} {student.records.length > 3 ? '⚠️' : ''}</b><br/><span>{student.id}</span></div>
                <div className="count-badge" onClick={(e) => { const el = e.target.parentElement.nextElementSibling; el.style.display = el.style.display === 'block' ? 'none' : 'block'; }}>{student.records.length}</div>
              </div>
              <div className="timestamp-list">{student.records.map((r, i) => <div key={i}>🕒 {r.time}</div>)}</div>
            </li>
          ))}
        </ul>
      </div>

      <div id="main-content">
        <div id="monitor-controls">
          <button className={`ctrl-btn ${showSettings ? 'active' : ''}`} onClick={() => { setShowSettings(!showSettings); setShowLogs(false); }}>⚙️ Settings</button>
          <button className={`ctrl-btn ${showLogs ? 'active' : ''}`} onClick={() => { setShowLogs(!showLogs); setShowSettings(false); }}>📊 Live Log</button>
        </div>
        <div id="visual-wrapper">
          <img src="1.jpg" id="monitor-image" className="side-img" alt="Monitor" />
          <div id="center-stage">
            <div id="placeholder" style={{textAlign:'center'}}><h1 style={{color:'white'}}>{statusText}</h1><p style={{color:'#aaa'}}>Please enter your Name and ID to start.</p></div>
            <div id="status-panel" style={{display: 'none'}}>
              <div id="status-audio" className={`status-box ${audioStatus.label === 'korean' && audioStatus.score > settings.confidence/100 ? 'active-red' : ''}`}>🎤 {audioStatus.label.toUpperCase()} ({Math.round(audioStatus.score * 100)}%)</div>
              <div id="status-video" className={`status-box ${videoStatus.state === 'Speaking' ? 'active-green' : ''}`}>{videoStatus.state === 'Speaking' ? '🗣️' : '🤐'} {Math.round(videoStatus.gap)}%</div>
            </div>
            <div id="camera-wrapper"><canvas ref={canvasRef} id="output_canvas"></canvas><div id="alert-overlay">🚨 DETECTED!</div></div>
            <div id="panel-settings" style={{ display: showSettings ? 'flex' : 'none' }} className="overlay-panel">
              <div className="panel-header"><span>SETTINGS</span><span style={{cursor:'pointer'}} onClick={() => setShowSettings(false)}>✕</span></div>
              <div className="setting-row"><div className="setting-label"><span>AI Confidence</span><span>{settings.confidence}%</span></div><input type="range" min="1" max="99" value={settings.confidence} onChange={(e) => setSettings({...settings, confidence: parseInt(e.target.value)})} /></div>
              <div className="setting-row"><div className="setting-label"><span>Mouth Open (Gap)</span><span>{(settings.mouthOpen / 10).toFixed(1)}%</span></div><input type="range" min="1" max="50" value={settings.mouthOpen} onChange={(e) => setSettings({...settings, mouthOpen: parseInt(e.target.value)})} /></div>
              <div className="setting-row"><div className="setting-label"><span>Lip Movement</span><span>Lv {settings.lipMovement}</span></div><input type="range" min="1" max="100" value={settings.lipMovement} onChange={(e) => setSettings({...settings, lipMovement: parseInt(e.target.value)})} /></div>
              <div className="setting-row"><div className="setting-label"><span>Strictness</span><span>{settings.strictness} frames</span></div><input type="range" min="1" max="10" value={settings.strictness} onChange={(e) => setSettings({...settings, strictness: parseInt(e.target.value)})} /></div>
            </div>
            <div id="panel-logs" style={{ display: showLogs ? 'flex' : 'none' }} className="overlay-panel">
              <div className="panel-header"><span>LIVE LOGS</span><span style={{cursor:'pointer'}} onClick={() => setShowLogs(false)}>✕</span></div>
              <div id="log-container">
                {logs.length === 0 && <div style={{textAlign:'center', color:'#555', marginTop:'20px'}}>Waiting for data...</div>}
                {logs.map(log => (<div key={log.id} className={`log-entry ${log.label === 'VIOLATION' ? 'violation' : ''}`}><span>[{log.time}] {log.label.toUpperCase()}</span><span>{log.visual} / {log.score}%</span></div>))}
              </div>
            </div>
          </div>
        </div>
      </div>
      <video id="input_video" playsInline style={{display:'none'}}></video>
    </>
  );
}
export default App;
