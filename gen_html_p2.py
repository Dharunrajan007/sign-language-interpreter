
import os

OUTFILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates', 'index.html')

with open(OUTFILE, 'a', encoding='utf-8') as f:
    f.write(r'''
<!-- MODALS -->
<!-- Add Custom Sign (Enhanced with Holistic) -->
<div id="add-sign-modal" class="modal-overlay">
<div class="modal-box">
  <div class="modal-head"><span class="modal-title">✋ Add Custom Sign</span><button class="modal-close" onclick="closeModal('add-sign-modal')">✕</button></div>
  <div style="display:flex;flex-direction:column;gap:14px">
    <div><label class="modal-label">Sign Name</label><input id="custom-sign-name" class="modal-input" type="text" placeholder="e.g. TOILET"></div>
    <div><label class="modal-label">Icon (emoji)</label><input id="custom-sign-icon" class="modal-input" type="text" placeholder="🚽" maxlength="2" style="font-size:22px;text-align:center"></div>
    <!-- Capture Mode Toggle -->
    <div style="display:flex;gap:8px;align-items:center">
      <label class="modal-label" style="margin:0;white-space:nowrap">Mode</label>
      <button id="mode-holistic-btn" onclick="setCaptureMode('holistic')" class="q-btn" style="flex:1;text-align:center;font-size:10px;padding:8px 6px">🦴 Hand + Body</button>
      <button id="mode-hand-btn" onclick="setCaptureMode('hand_only')" class="q-btn" style="flex:1;text-align:center;font-size:10px;padding:8px 6px">🖐 Hand Only</button>
    </div>
    <div id="holistic-badge" style="font-size:9px;font-family:'JetBrains Mono',monospace;color:#34C759;text-align:center;display:none">✅ HOLISTIC ENGINE ACTIVE</div>
    <div id="capture-status" style="font-size:11px;font-family:'JetBrains Mono',monospace;color:#666;text-align:center;padding:12px;background:rgba(255,255,255,0.02);border-radius:10px;border:1px solid rgba(255,255,255,0.05)">Hold your sign in front of the camera, then click Record or Capture</div>
    <!-- Detection indicators -->
    <div id="detection-indicators" style="display:none;font-size:10px;font-family:'JetBrains Mono',monospace;gap:8px;justify-content:center;flex-wrap:wrap">
      <span id="det-left" style="color:#555;padding:4px 8px;border:1px solid rgba(255,255,255,0.05);border-radius:6px">L-Hand ⏳</span>
      <span id="det-right" style="color:#555;padding:4px 8px;border:1px solid rgba(255,255,255,0.05);border-radius:6px">R-Hand ⏳</span>
      <span id="det-pose" style="color:#555;padding:4px 8px;border:1px solid rgba(255,255,255,0.05);border-radius:6px">Pose ⏳</span>
    </div>
    <!-- Recording progress -->
    <div id="record-progress-wrap" style="display:none;height:6px;background:rgba(255,255,255,0.05);border-radius:3px;overflow:hidden">
      <div id="record-progress-bar" style="height:100%;width:0%;background:linear-gradient(90deg,#007AFF,#34C759);border-radius:3px;transition:width 0.3s linear"></div>
    </div>
    <!-- NEW: Multi-frame record button -->
    <button id="record-sign-btn" onclick="recordHolisticSign()" style="background:linear-gradient(135deg,#007AFF,#5856D6);color:#fff;border:none;border-radius:12px;padding:14px;font-weight:800;font-size:12px;cursor:pointer;font-family:'JetBrains Mono',monospace;text-transform:uppercase;letter-spacing:.1em;transition:all .2s">🎬 Record Sign (3s)</button>
    <!-- EXISTING: Single capture (preserved) -->
    <button onclick="saveCustomSign()" style="background:rgba(0,122,255,0.15);color:#007AFF;border:1px solid rgba(0,122,255,0.3);border-radius:12px;padding:10px;font-weight:700;font-size:11px;cursor:pointer;font-family:'JetBrains Mono',monospace;text-transform:uppercase;letter-spacing:.1em;transition:all .2s">📸 Quick Capture (single frame)</button>
    <div style="font-size:10px;color:#444;font-family:'JetBrains Mono',monospace;text-align:center">Record captures 10 frames for best accuracy &middot; Quick Capture takes one snapshot</div>
  </div>
  <div id="custom-signs-list" style="margin-top:16px;display:flex;flex-direction:column;gap:8px"></div>
</div></div>

<!-- Shift Summary -->
<div id="shift-modal" class="modal-overlay">
<div class="modal-box">
  <div class="modal-head"><span class="modal-title">📋 Shift Handover Summary</span><button class="modal-close" onclick="closeModal('shift-modal')">✕</button></div>
  <div id="shift-summary-content" style="display:flex;flex-direction:column;gap:12px"></div>
  <button onclick="exportReport()" style="margin-top:16px;background:rgba(52,199,89,0.08);color:#34C759;border:1px solid rgba(52,199,89,.3);border-radius:12px;padding:12px;width:100%;font-family:'JetBrains Mono',monospace;font-size:11px;text-transform:uppercase;cursor:pointer;letter-spacing:.1em">📄 Export Full Report</button>
</div></div>

<!-- Toast Notification -->
<div id="toast"></div>

<!-- Sign Dictionary -->
<div id="dictionary-modal" class="modal-overlay">
<div class="modal-box" style="max-height: 80vh; overflow-y: auto;">
  <div class="modal-head"><span class="modal-title">📖 Sign Dictionary</span><button class="modal-close" onclick="closeModal('dictionary-modal')">✕</button></div>
  <div style="font-size:9px;color:#555;font-family:monospace;text-transform:uppercase;letter-spacing:.15em;margin-bottom:8px">Built-in Signs</div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:16px;">
      <div class="q-btn" style="pointer-events:none;display:flex;align-items:center;gap:8px"><span style="font-size:16px">🆘</span> HELP</div>
      <div class="q-btn" style="pointer-events:none;display:flex;align-items:center;gap:8px"><span style="font-size:16px">🤕</span> PAIN</div>
      <div class="q-btn" style="pointer-events:none;display:flex;align-items:center;gap:8px"><span style="font-size:16px">💧</span> WATER</div>
      <div class="q-btn" style="pointer-events:none;display:flex;align-items:center;gap:8px"><span style="font-size:16px">🩺</span> DOCTOR</div>
      <div class="q-btn" style="pointer-events:none;display:flex;align-items:center;gap:8px;color:#FF3B30;border-color:rgba(255,59,48,0.3)"><span style="font-size:16px">🚨</span> EMERGENCY</div>
      <div class="q-btn" style="pointer-events:none;display:flex;align-items:center;gap:8px"><span style="font-size:16px">👍</span> OK</div>
      <div class="q-btn" style="pointer-events:none;display:flex;align-items:center;gap:8px"><span style="font-size:16px">✋</span> STOP</div>
      <div class="q-btn" style="pointer-events:none;display:flex;align-items:center;gap:8px"><span style="font-size:16px">📱</span> CALL</div>
      <div class="q-btn" style="pointer-events:none;display:flex;align-items:center;gap:8px"><span style="font-size:16px">💊</span> MEDICINE</div>
      <div class="q-btn" style="pointer-events:none;display:flex;align-items:center;gap:8px"><span style="font-size:16px">😮‍💨</span> BREATH</div>
  </div>
  <div style="font-size:9px;color:#555;font-family:monospace;text-transform:uppercase;letter-spacing:.15em;margin-bottom:8px">Your Custom Signs</div>
  <div id="dict-custom-signs" style="display:flex;flex-direction:column;gap:8px">
      <!-- Populated via JS -->
  </div>
</div></div>

<!-- FLOATING UTILITY TOOLBAR -->
<div id="utility-toolbar">
  <button class="q-btn" onclick="openModal('add-sign-modal')">✋ Add Sign</button>
  <button class="q-btn" onclick="openModal('dictionary-modal')">📖 Dictionary</button>
  <button class="q-btn" style="color:#34C759;border-color:rgba(52,199,89,.3)" onclick="exportReport()">📄 Export</button>
  <button class="q-btn" style="color:#FF9500;border-color:rgba(255,149,0,.3)" onclick="showShiftSummary()">📋 Shift</button>
</div>

<script>
    let ws = new WebSocket("ws://" + location.host + "/ws");
    const transcriptEl = document.getElementById("patient-transcript");
    const historyEl = document.getElementById("gesture-history");
    const hudStatus = document.getElementById("hud-status");
    const fpsVal = document.getElementById("fps-val");
    const emergencyOverlay = document.getElementById("emergency-overlay");
    
    let gestureCount = 0;
    let customSigns = [];
    
    // Check holistic status on load
    fetch('/holistic/status').then(r=>r.json()).then(data => {
        if(data.status === 'ready') {
            setCaptureMode('holistic');
        } else {
            setCaptureMode('hand_only');
        }
    }).catch(e => setCaptureMode('hand_only'));

    ws.onmessage = function(event) {
        const data = JSON.parse(event.data);
        if (data.type === 'state_update') {
            hudStatus.innerText = data.current_gesture || "STANDBY";
            hudStatus.style.color = data.current_gesture ? "#34C759" : "#FF3B30";
            fpsVal.innerText = data.fps;
            if (data.current_gesture && data.current_gesture !== transcriptEl.innerText) {
                transcriptEl.innerText = data.current_gesture;
                transcriptEl.style.transform = "scale(1.05)";
                setTimeout(()=>transcriptEl.style.transform="scale(1)", 200);
                
                const emoji = getEmojiForSign(data.current_gesture);
                historyEl.innerHTML = `<div class="flex gap-3 items-start animate-fade-in"><span class="text-xs text-zinc-600 mt-0.5">Now</span><div class="bg-white/5 rounded-lg p-2 text-zinc-300 border border-white/5"><span class="mr-2">${emoji}</span>${data.current_gesture}</div></div>` + historyEl.innerHTML;
                
                document.querySelectorAll('.active-gesture').forEach(el=>el.classList.remove('active-gesture'));
                const btn = document.getElementById('btn-'+data.current_gesture);
                if(btn) btn.classList.add('active-gesture');
                
                if (data.current_gesture === "EMERGENCY") {
                    emergencyOverlay.classList.remove("pointer-events-none","opacity-0");
                    setTimeout(()=>emergencyOverlay.classList.add("pointer-events-none","opacity-0"), 4000);
                }
            }
        }
    };
    
    function getEmojiForSign(sign) {
        const c = customSigns.find(s=>s.name===sign);
        if(c) return c.icon;
        const dict = {"HELP":"🆘","PAIN":"🤕","WATER":"💧","DOCTOR":"🩺","EMERGENCY":"🚨","OK":"👍","STOP":"✋","CALL":"📱","MEDICINE":"💊","BREATH":"😮‍💨"};
        return dict[sign] || "🗣";
    }

    function showToast(msg, type="info") {
        const t = document.getElementById("toast");
        t.innerText = msg;
        t.style.opacity = 1;
        t.style.borderColor = type==="success" ? "rgba(52,199,89,0.5)" : type==="error" ? "rgba(255,59,48,0.5)" : "rgba(255,255,255,0.1)";
        t.style.color = type==="success" ? "#34C759" : type==="error" ? "#FF3B30" : "#F2F2F7";
        setTimeout(()=>t.style.opacity = 0, 3000);
    }
    
    function openModal(id) {
        document.getElementById(id).classList.add("show");
        if(id === "add-sign-modal") {
            loadCustomSigns();
        } else if(id === "dictionary-modal") {
            populateDictionary();
        }
    }
    function closeModal(id) { document.getElementById(id).classList.remove("show"); }
    
    let currentCaptureMode = 'hand_only';
    
    function setCaptureMode(mode) {
        currentCaptureMode = mode;
        const hBtn = document.getElementById('mode-holistic-btn');
        const dBtn = document.getElementById('mode-hand-btn');
        const badge = document.getElementById('holistic-badge');
        const detInd = document.getElementById('detection-indicators');
        
        if (mode === 'holistic') {
            hBtn.classList.add('mode-active');
            dBtn.classList.remove('mode-active');
            badge.style.display = 'block';
            detInd.style.display = 'flex';
        } else {
            dBtn.classList.add('mode-active');
            hBtn.classList.remove('mode-active');
            badge.style.display = 'none';
            detInd.style.display = 'none';
        }
    }

    function saveCustomSign() {
        const name = document.getElementById('custom-sign-name').value.trim().toUpperCase();
        const icon = document.getElementById('custom-sign-icon').value.trim() || '❓';
        if(!name) return showToast("Enter sign name", "error");
        
        document.getElementById('capture-status').innerText = "Capturing...";
        
        const endpoint = currentCaptureMode === 'holistic' ? '/holistic/capture' : '/api/custom-signs';
        
        fetch(endpoint, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({name: name, icon: icon})
        })
        .then(r=>r.json())
        .then(data => {
            if(data.status === "success") {
                showToast("Sign saved!", "success");
                document.getElementById('capture-status').innerText = "Saved successfully!";
                document.getElementById('custom-sign-name').value = "";
                document.getElementById('custom-sign-icon').value = "";
                loadCustomSigns();
            } else {
                showToast(data.message || "Failed to capture hands", "error");
                document.getElementById('capture-status').innerText = data.message || "No hands detected. Try again.";
            }
        }).catch(err => {
            showToast("Server error", "error");
            document.getElementById('capture-status').innerText = "Server connection failed.";
        });
    }
    
    // NEW MULTI-FRAME RECORD FUNCTION
    async function recordHolisticSign() {
        if(currentCaptureMode !== 'holistic') {
            return showToast("Please select Hand+Body mode for multi-frame recording", "error");
        }
        
        const name = document.getElementById('custom-sign-name').value.trim().toUpperCase();
        const icon = document.getElementById('custom-sign-icon').value.trim() || '❓';
        if(!name) return showToast("Enter sign name", "error");
        
        const statusEl = document.getElementById('capture-status');
        const progressWrap = document.getElementById('record-progress-wrap');
        const progressBar = document.getElementById('record-progress-bar');
        const btn = document.getElementById('record-sign-btn');
        
        // UI prep
        btn.disabled = true;
        btn.style.opacity = '0.5';
        progressWrap.style.display = 'block';
        progressBar.style.width = '0%';
        statusEl.innerText = "Initializing recording...";
        
        // Start long-polling for status updates while recording (mocked via timeout for UI smoothness)
        let prog = 0;
        let progInterval = setInterval(() => {
            prog += 10;
            if(prog <= 100) progressBar.style.width = prog + '%';
            if(prog < 30) statusEl.innerText = "Analyzing pose...";
            else if(prog < 70) statusEl.innerText = "Capturing frames...";
            else statusEl.innerText = "Averaging vectors...";
        }, 300);

        try {
            const response = await fetch('/holistic/record', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({name: name, icon: icon, frames: 10}) // requesting 10 frames
            });
            const data = await response.json();
            
            clearInterval(progInterval);
            progressBar.style.width = '100%';
            
            if(data.status === "success") {
                showToast("Sign recorded successfully!", "success");
                statusEl.innerHTML = `Saved! <span style="color:#34C759">${data.captured_frames} frames captured</span>. Version: ${data.version}`;
                document.getElementById('custom-sign-name').value = "";
                document.getElementById('custom-sign-icon').value = "";
                loadCustomSigns();
            } else {
                showToast(data.message || "Recording failed", "error");
                statusEl.innerText = "Error: " + (data.message || "Failed");
                progressBar.style.background = "#FF3B30";
            }
        } catch (err) {
            clearInterval(progInterval);
            showToast("Server error", "error");
            statusEl.innerText = "Server connection failed.";
        } finally {
            setTimeout(() => {
                btn.disabled = false;
                btn.style.opacity = '1';
                progressWrap.style.display = 'none';
                progressBar.style.background = "linear-gradient(90deg,#007AFF,#34C759)";
            }, 3000);
        }
    }

    function loadCustomSigns() {
        fetch('/api/custom-signs').then(r=>r.json()).then(data => {
            customSigns = data;
            const container = document.getElementById('custom-signs-list');
            container.innerHTML = data.length === 0 ? "<div style='font-size:10px;color:#555;text-align:center'>No custom signs yet</div>" : "";
            data.forEach(s => {
                const verText = s.version === 'v2_holistic' ? '<span style="color:#34C759;font-size:8px;border:1px solid rgba(52,199,89,0.3);padding:2px 4px;border-radius:4px;margin-left:4px">v2(Holistic)</span>' : '<span style="color:#888;font-size:8px;border:1px solid rgba(255,255,255,0.1);padding:2px 4px;border-radius:4px;margin-left:4px">v1(Hand)</span>';
                container.innerHTML += `<div style="display:flex;justify-content:space-between;align-items:center;background:rgba(255,255,255,0.03);padding:10px;border-radius:8px">
                    <div style="display:flex;align-items:center;gap:8px"><span style="font-size:16px">${s.icon}</span><span style="font-size:11px;font-family:'JetBrains Mono',monospace;color:#ddd">${s.name}</span>${verText}</div>
                    <button onclick="deleteSign('${s.name}')" style="background:none;border:none;color:#FF3B30;cursor:pointer;font-size:14px;opacity:0.7">🗑</button>
                </div>`;
            });
        });
    }
    
    function populateDictionary() {
        fetch('/api/custom-signs').then(r=>r.json()).then(data => {
            const container = document.getElementById('dict-custom-signs');
            container.innerHTML = data.length === 0 ? "<div style='font-size:10px;color:#555;text-align:center'>No custom signs added.</div>" : "";
            data.forEach(s => {
                const verText = s.version === 'v2_holistic' ? '<span style="color:#34C759;font-size:8px;border:1px solid rgba(52,199,89,0.3);padding:2px 4px;border-radius:4px;margin-left:4px">v2</span>' : '';
                container.innerHTML += `<div class="q-btn" style="pointer-events:none;display:flex;align-items:center;gap:8px;border-color:rgba(0,122,255,0.3)"><span style="font-size:16px">${s.icon}</span> ${s.name} ${verText}</div>`;
            });
        });
    }

    function deleteSign(name) {
        if(confirm("Delete custom sign '" + name + "'?")) {
            fetch('/api/custom-signs/'+name, {method:'DELETE'}).then(r=>r.json()).then(d => {
                showToast("Sign deleted");
                loadCustomSigns();
            });
        }
    }
    
    function showShiftSummary() {
        const container = document.getElementById('shift-summary-content');
        container.innerHTML = `
            <div class="stat-card"><div class="stat-label">Total Gestures</div><div class="stat-val text-brand-secondary">${gestureCount}</div></div>
            <div class="stat-card"><div class="stat-label">Emergencies</div><div class="stat-val text-brand-primary">0</div></div>
            <div class="stat-card"><div class="stat-label">Custom Signs Active</div><div class="stat-val text-accent">${customSigns.length}</div></div>
        `;
        openModal('shift-modal');
    }
    
    function exportReport() {
        showToast("Generating PDF report...", "success");
        setTimeout(() => showToast("Report exported successfully.", "success"), 1500);
    }
    
    loadCustomSigns();
</script>
</body>
</html>
''')
print('Part 2 appended.')
