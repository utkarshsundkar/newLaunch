/**
 * Sample React Native App
 * https://github.com/facebook/react-native
 *
 * @format
 */

import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { ActivityIndicator, StatusBar, StyleSheet, Text, View, TouchableOpacity, Image as RNImage, ScrollView } from 'react-native';
import { Camera, useCameraDevice, CameraDevice } from 'react-native-vision-camera';
import { Image as CompressorImage } from 'react-native-compressor';
import { Canvas, Circle, Line } from '@shopify/react-native-skia';
import RNFS from 'react-native-fs';
import {launchImageLibrary, ImageLibraryOptions, Asset} from 'react-native-image-picker';

export default function App() {
  const [hasPermission, setHasPermission] = useState<boolean | null>(null);
  const [device, setDevice] = useState<CameraDevice | undefined>(undefined);
  const [previewSize, setPreviewSize] = useState<{ width: number; height: number } | null>(null);
  const cameraRef = useRef<Camera>(null);
  const [uploadMode, setUploadMode] = useState(false);
  const [analyze, setAnalyze] = useState<any>(null);
  const [pushupData, setPushupData] = useState<any>(null);
  // Add state for diamond pushup data
  const [diamondPushupData, setDiamondPushupData] = useState<any>(null);
  const [analyzeError, setAnalyzeError] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);
  const [reps, setReps] = useState(0);
  const [phase, setPhase] = useState<'INIT' | 'OPEN' | 'CLOSE' | 'UP' | 'DOWN' | 'MID' | 'COMPLETE'>('INIT');
  const [serverFps, setServerFps] = useState<number | null>(null);
  const [landmarks, setLandmarks] = useState<any>(null);
  const [feetApart, setFeetApart] = useState<number | null>(null);
  const [fingers, setFingers] = useState<number | null>(null);
  const [thresholds, setThresholds] = useState({ open_feet: 1.8, close_feet: 0.8, open_hands: 0.15, close_hands: 0.15 });
  const [exerciseType, setExerciseType] = useState<'jacks' | 'pushups' | 'diamond_pushups' | null>(null);
  const [msgCount, setMsgCount] = useState(0);
  const [wsConnected, setWsConnected] = useState(false);

  useEffect(() => {
    (async () => {
      const status = await Camera.requestCameraPermission();
      setHasPermission(status === 'granted');
      const devices = await Camera.getAvailableCameraDevices();
      const backDevice = devices.find((d) => d.position === 'back');
      if (backDevice) {
        setDevice(backDevice);
      }
    })();
  }, []);

  const processServerResult = useCallback((result: any) => {
    if (!result) return;
    // Diamond pushup response handling (check this first since it has more specific fields)
    if (typeof result?.hand_distance === 'number') {
      setDiamondPushupData(result);
      setReps(result.reps || 0);
      if (typeof result?.phase === 'string') setPhase(result.phase as any);
      setServerFps(typeof result?.fps === 'number' ? result.fps : null);
      if (result?.landmarks) setLandmarks(result.landmarks);
      setFingers(null);
      setAnalyze(null); // Clear jumping jacks analysis
      setPushupData(null); // Clear pushup data
      setExerciseType('diamond_pushups'); // Set exercise type for live tracking
      return;
    }
    // Pushup response handling (check this first since it has more specific fields)
    if (typeof result?.body_angle === 'number') {
      setPushupData(result);
      setReps(result.reps || 0);
      if (typeof result?.phase === 'string') setPhase(result.phase as any);
      setServerFps(typeof result?.fps === 'number' ? result.fps : null);
      if (result?.landmarks) setLandmarks(result.landmarks);
      setFingers(null);
      setAnalyze(null); // Clear jumping jacks analysis
      setDiamondPushupData(null); // Clear diamond pushup data
      setExerciseType('pushups'); // Set exercise type for live tracking
      return;
    }
    // Server-side Jacks response has reps/phase
    if (typeof result?.reps === 'number') {
      setReps(result.reps);
      if (typeof result?.phase === 'string') setPhase(result.phase as any);
      setServerFps(typeof result?.fps === 'number' ? result.fps : null);
      if (result?.landmarks) setLandmarks(result.landmarks);
      setFingers(null);
      setPushupData(null); // Clear pushup data
      setDiamondPushupData(null); // Clear diamond pushup data
      setExerciseType('jacks'); // Set exercise type for live tracking
      return;
    }
    // Fingers test response (fallback)
    if (typeof result?.fingers === 'number') {
      setFingers(result.fingers);
      setServerFps(typeof result?.fps === 'number' ? result.fps : null);
      setPushupData(null); // Clear pushup data
      setDiamondPushupData(null); // Clear diamond pushup data
      setAnalyze(null); // Clear jumping jacks analysis
      setExerciseType(null); // Clear exercise type
      return;
    }
  }, [thresholds]);

  // Server URL: switch to /jacks for server-side rep counting
  const wsUrl = useMemo(() => exerciseType === 'diamond_pushups'
    ? `ws://192.168.0.105:8001/ws/diamond_pushups?flip=1&session=default`
    : exerciseType === 'pushups' 
    ? `ws://192.168.0.105:8001/ws/pushups?flip=1&session=default` 
    : 'ws://192.168.0.105:8001/ws/jacks?flip=1&session=default', [exerciseType]);
  const serverUrl = useMemo(() => exerciseType === 'diamond_pushups'
    ? 'http://192.168.0.105:8001/diamond_pushups'
    : exerciseType === 'pushups'
    ? 'http://192.168.0.105:8001/pushups'
    : 'http://192.168.0.105:8001/jacks', [exerciseType]);
  const analyzeUrl = useMemo(() => 'http://192.168.0.105:8001/analyze', []);
  // Add pushup analysis URL
  const pushupAnalyzeUrl = useMemo(() => 'http://192.168.0.105:8001/pushups', []);
  // Add diamond pushup analysis URL
  const diamondPushupAnalyzeUrl = useMemo(() => 'http://192.168.0.105:8001/diamond_pushups', []);
  const wsRef = useRef<WebSocket | null>(null);
  const inflightRef = useRef<boolean>(false);

  const pickAndUpload = useCallback(async () => {
    try {
      setAnalyze(null);
      setAnalyzeError(null);
      setPushupData(null); // Clear pushup data
      setExerciseType('jacks'); // Set exercise type
      setUploading(true);
      const options: ImageLibraryOptions = {mediaType: 'video', selectionLimit: 1};
      const result = await launchImageLibrary(options);
      if (result.didCancel) return;
      const asset: Asset | undefined = result.assets && result.assets[0];
      if (!asset?.uri) {
        setAnalyzeError('No video selected');
        return;
      }
      const form = new FormData();
      form.append('file', {
        // @ts-ignore
        uri: asset.uri,
        type: asset.type || 'video/mp4',
        name: asset.fileName || 'video.mp4',
      });
      form.append('flip', '1');
      // Use the appropriate endpoint based on the exercise type
      // For now, we'll keep the existing jumping jacks analysis
      const res = await fetch(analyzeUrl, { method: 'POST', headers: { Accept: 'application/json' }, body: form });
      if (!res.ok) {
        const txt = await res.text();
        setAnalyzeError(`Upload failed: ${res.status} ${txt}`);
        return;
      }
      const json = await res.json();
      setAnalyze(json);
    } catch (e: any) {
      setAnalyzeError(String(e?.message || e));
    } finally {
      setUploading(false);
    }
  }, [analyzeUrl]);

  // Add a new function for pushup analysis
  const pickAndUploadPushups = useCallback(async () => {
    try {
      setAnalyze(null);
      setAnalyzeError(null);
      setPushupData(null); // Clear previous pushup data
      setDiamondPushupData(null); // Clear diamond pushup data
      setExerciseType('pushups'); // Set exercise type
      setUploading(true);
      const options: ImageLibraryOptions = {mediaType: 'video', selectionLimit: 1};
      const result = await launchImageLibrary(options);
      if (result.didCancel) return;
      const asset: Asset | undefined = result.assets && result.assets[0];
      if (!asset?.uri) {
        setAnalyzeError('No video selected');
        return;
      }
      const form = new FormData();
      form.append('file', {
        // @ts-ignore
        uri: asset.uri,
        type: asset.type || 'video/mp4',
        name: asset.fileName || 'video.mp4',
      });
      form.append('flip', '1');
      // Use the pushup analysis endpoint
      const res = await fetch(pushupAnalyzeUrl, { method: 'POST', headers: { Accept: 'application/json' }, body: form });
      if (!res.ok) {
        const txt = await res.text();
        setAnalyzeError(`Upload failed: ${res.status} ${txt}`);
        return;
      }
      const json = await res.json();
      // For pushups, we set the pushupData directly instead of using the analyze state
      setPushupData(json);
    } catch (e: any) {
      setAnalyzeError(String(e?.message || e));
    } finally {
      setUploading(false);
    }
  }, [pushupAnalyzeUrl]);

  // Add a new function for diamond pushup analysis
  const pickAndUploadDiamondPushups = useCallback(async () => {
    try {
      setAnalyze(null);
      setAnalyzeError(null);
      setPushupData(null); // Clear pushup data
      setDiamondPushupData(null); // Clear previous diamond pushup data
      setExerciseType('diamond_pushups'); // Set exercise type
      setUploading(true);
      const options: ImageLibraryOptions = {mediaType: 'video', selectionLimit: 1};
      const result = await launchImageLibrary(options);
      if (result.didCancel) return;
      const asset: Asset | undefined = result.assets && result.assets[0];
      if (!asset?.uri) {
        setAnalyzeError('No video selected');
        return;
      }
      const form = new FormData();
      form.append('file', {
        // @ts-ignore
        uri: asset.uri,
        type: asset.type || 'video/mp4',
        name: asset.fileName || 'video.mp4',
      });
      form.append('flip', '1');
      // Use the diamond pushup analysis endpoint
      const res = await fetch(diamondPushupAnalyzeUrl, { method: 'POST', headers: { Accept: 'application/json' }, body: form });
      if (!res.ok) {
        const txt = await res.text();
        setAnalyzeError(`Upload failed: ${res.status} ${txt}`);
        return;
      }
      const json = await res.json();
      // For diamond pushups, we set the diamondPushupData directly
      setDiamondPushupData(json);
    } catch (e: any) {
      setAnalyzeError(String(e?.message || e));
    } finally {
      setUploading(false);
    }
  }, [diamondPushupAnalyzeUrl]);

  useEffect(() => {
    if (uploadMode) return; // disable live tracking when in upload mode
    if (!hasPermission) return;
    let timer: any;
    // open WS
    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;
    ws.onopen = () => { setWsConnected(true); };
    ws.onmessage = (ev) => {
      try {
        const parsed = JSON.parse(ev.data as string);
        setMsgCount((c) => c + 1);
        processServerResult(parsed);
        inflightRef.current = false; // allow next frame
      } catch (e) {
        // eslint-disable-next-line no-console
        console.log('ws parse error', e);
      }
    };
    ws.onerror = (e) => { setWsConnected(false); /* eslint-disable-next-line no-console */ console.log('ws error', e); };
    ws.onclose = () => { wsRef.current = null; setWsConnected(false); };

    // aggressive low-latency cadence
    const intervalMs = 90;
    const tick = async () => {
      if (uploading || inflightRef.current) return;
      const cam = cameraRef.current;
      if (!cam) return;
      try {
        setUploading(true);
        const photo = await cam.takePhoto({ flash: 'off' });
        const compressedPath = await CompressorImage.compress('file://' + photo.path, {
          compressionMethod: 'auto',
          quality: 0.25,
          maxWidth: 160,
        });
        const fileData = await RNFS.readFile(compressedPath.replace('file://',''), 'base64');
        if (wsRef.current && wsRef.current.readyState === 1) {
          inflightRef.current = true;
          wsRef.current.send(JSON.stringify({ jpg_b64: fileData }));
        } else {
          // HTTP fallback if WS is disconnected
          const form = new FormData();
          form.append('file', {
            // @ts-ignore
            uri: compressedPath,
            type: 'image/jpeg',
            name: 'frame.jpg',
          });
          form.append('flip', '1');
          form.append('session', 'default');
          // Use the appropriate endpoint based on exercise type
          const res = await fetch(serverUrl, { method: 'POST', headers: { Accept: 'application/json' }, body: form });
          if (res.ok) {
            const json = await res.json();
            processServerResult(json);
          }
        }
      } catch (e) {
        // eslint-disable-next-line no-console
        console.log('send error', e);
      } finally {
        setUploading(false);
      }
    };
    timer = setInterval(tick, intervalMs);
    return () => { clearInterval(timer); if (wsRef.current) wsRef.current.close(); };
  }, [hasPermission, wsUrl, uploading, processServerResult, serverUrl]);

  // If in upload mode, render upload UI instead of live tracking
  if (uploadMode) {
    return (
      <View style={{ flex: 1, backgroundColor: 'black', alignItems: 'center', justifyContent: 'center' }}>
        <StatusBar barStyle={'light-content'} />
        <Text style={{ color: 'white', fontSize: 20, marginBottom: 16 }}>Exercise Analyzer</Text>
        <View style={{ flexDirection: 'row', marginBottom: 16, flexWrap: 'wrap', justifyContent: 'center' }}>
          <TouchableOpacity disabled={uploading} onPress={pickAndUpload} style={{ paddingVertical: 12, paddingHorizontal: 20, backgroundColor: '#4CAF50', borderRadius: 8, marginRight: 10, marginBottom: 10 }}>
            <Text style={{ color: 'white', fontSize: 16 }}>{uploading ? 'Analyzing…' : 'Analyze Jumping Jacks'}</Text>
          </TouchableOpacity>
          <TouchableOpacity disabled={uploading} onPress={pickAndUploadPushups} style={{ paddingVertical: 12, paddingHorizontal: 20, backgroundColor: '#2196F3', borderRadius: 8, marginRight: 10, marginBottom: 10 }}>
            <Text style={{ color: 'white', fontSize: 16 }}>{uploading ? 'Analyzing…' : 'Analyze Pushups'}</Text>
          </TouchableOpacity>
          <TouchableOpacity disabled={uploading} onPress={pickAndUploadDiamondPushups} style={{ paddingVertical: 12, paddingHorizontal: 20, backgroundColor: '#9C27B0', borderRadius: 8, marginBottom: 10 }}>
            <Text style={{ color: 'white', fontSize: 16 }}>{uploading ? 'Analyzing…' : 'Analyze Diamond Pushups'}</Text>
          </TouchableOpacity>
        </View>
        {exerciseType === 'jacks' && analyze && (
          <View style={{ marginTop: 20, alignItems: 'center' }}>
            <Text style={{ color: 'white', fontSize: 18 }}>Reps: {analyze.reps}</Text>
            {'reps_perfect' in analyze && 'reps_wrong' in analyze && (
              <Text style={{ color: 'white', marginTop: 6 }}>Perfect: {/** @ts-ignore */ analyze.reps_perfect}  •  Wrong: {/** @ts-ignore */ analyze.reps_wrong}</Text>
            )}
            <Text style={{ color: 'white', marginTop: 6 }}>Frames: {analyze.frames}  •  Duration: {analyze.duration_s.toFixed(1)}s  •  Proc FPS: {analyze.processed_fps.toFixed(1)}</Text>
            {Array.isArray(analyze.diagnostics) && analyze.diagnostics.length > 0 && (
              <ScrollView style={{ maxHeight: 280, marginTop: 16 }} contentContainerStyle={{ paddingBottom: 20 }}>
                {analyze.diagnostics!.map((d: any, i: number) => (
                  <View key={`diag-${i}`} style={{ marginBottom: 16, alignItems: 'center' }}>
                    <Text style={{ color: 'tomato', marginBottom: 6 }}>Rep {d.rep_index}: {d.description}</Text>
                    {d.image_b64 ? (
                      <RNImage
                        source={{ uri: `data:image/jpeg;base64,${d.image_b64}` }}
                        resizeMode="contain"
                        style={{ width: 280, height: 180, borderRadius: 8, borderWidth: 1, borderColor: '#444' }}
                      />
                    ) : null}
                  </View>
                ))}
              </ScrollView>
            )}
          </View>
        )}
        {exerciseType === 'pushups' && pushupData && (
          <View style={{ marginTop: 20, alignItems: 'center' }}>
            <Text style={{ color: 'white', fontSize: 18 }}>Pushup Reps: {pushupData.reps}</Text>
            <Text style={{ color: 'white', marginTop: 6 }}>Phase: {pushupData.phase}</Text>
            <Text style={{ color: 'white', marginTop: 6 }}>Body Angle: {pushupData.body_angle?.toFixed(2)}</Text>
            <Text style={{ color: 'white', marginTop: 6 }}>Arms Angle: {pushupData.arms_angle?.toFixed(2)}</Text>
            <Text style={{ color: 'white', marginTop: 6 }}>Position: {pushupData.is_down ? 'Down' : pushupData.is_up ? 'Up' : 'Mid'}</Text>
            <Text style={{ color: 'white', marginTop: 6 }}>FPS: {pushupData.fps?.toFixed(1)}</Text>
          </View>
        )}
        {exerciseType === 'diamond_pushups' && diamondPushupData && (
          <View style={{ marginTop: 20, alignItems: 'center' }}>
            <Text style={{ color: 'white', fontSize: 18, fontWeight: 'bold' }}>Diamond Pushup Reps: {diamondPushupData.reps}</Text>
            <Text style={{ color: 'white', marginTop: 6 }}>Phase: {diamondPushupData.phase}</Text>
            <Text style={{ color: 'white', marginTop: 6 }}>Body Angle: {diamondPushupData.body_angle?.toFixed(2)}</Text>
            <Text style={{ color: 'white', marginTop: 6 }}>Arms Angle: {diamondPushupData.arms_angle?.toFixed(2)}</Text>
            <Text style={{ color: 'white', marginTop: 6 }}>Hand Distance: {diamondPushupData.hand_distance?.toFixed(2)}</Text>
            <Text style={{ color: 'white', marginTop: 6 }}>Position: {diamondPushupData.is_down ? 'Down' : diamondPushupData.is_up ? 'Up' : 'Mid'}</Text>
            <Text style={{ color: 'white', marginTop: 6 }}>FPS: {diamondPushupData.fps?.toFixed(1)}</Text>
          </View>
        )}
        {analyzeError && (
          <Text style={{ color: 'tomato', marginTop: 12 }}>{analyzeError}</Text>
        )}
      </View>
    );
  }

  // Add a function to toggle between exercise types in live tracking
  const toggleExerciseType = () => {
    if (exerciseType === 'jacks') {
      setExerciseType('pushups');
    } else if (exerciseType === 'pushups') {
      setExerciseType('diamond_pushups');
    } else {
      setExerciseType('jacks');
    }
  };

  const overlay = useMemo(() => {
    const isFront = device?.position === 'front';
    const mapPt = (pt?: { x: number; y: number }) => {
      if (!pt || !previewSize) return { x: 0, y: 0 };
      const nx = Math.max(0, Math.min(1, pt.x));
      const ny = Math.max(0, Math.min(1, pt.y));
      const x = (isFront ? 1 - nx : nx) * previewSize.width;
      const y = ny * previewSize.height;
      return { x, y };
    };

    const dot = (p?: { x: number; y: number }, key?: string) => {
      const m = mapPt(p);
      return <Circle key={`d-${key}`} cx={m.x} cy={m.y} r={4} color="#00e5ff" />;
    };

    const seg = (a?: { x: number; y: number }, b?: { x: number; y: number }, key?: string, color = '#00e5ff') => {
      const p1 = mapPt(a);
      const p2 = mapPt(b);
      return <Line key={`l-${key}`} p1={p1} p2={p2} color={color} strokeWidth={3} />;
    };

    const L = landmarks as any;

    return (
      <View pointerEvents="none" style={StyleSheet.absoluteFill}>
        {/* HUD */}
        <View style={styles.overlayBox}>
          <Text style={styles.overlayText}>Reps: {reps}  •  State: {phase}</Text>
          <Text style={styles.overlaySub}>
            {wsConnected ? 'WS: connected' : 'WS: disconnected'}  {`Msgs: ${msgCount}`}
            {serverFps ? `  • Srv FPS: ${serverFps.toFixed(1)}` : ''}
            {feetApart ? `  • Feet: ${feetApart.toFixed(2)}` : ''}
            {exerciseType === 'pushups' && pushupData && (
              `  • Body: ${pushupData.body_angle?.toFixed(2)}  • Arms: ${pushupData.arms_angle?.toFixed(2)}`
            )}
            {exerciseType === 'diamond_pushups' && diamondPushupData && (
              `  • Body: ${diamondPushupData.body_angle?.toFixed(2)}  • Arms: ${diamondPushupData.arms_angle?.toFixed(2)}  • Hands: ${diamondPushupData.hand_distance?.toFixed(2)}`
            )}
          </Text>
        </View>
        {/* Fingers big counter (server-side) */}
        {typeof fingers === 'number' && (
          <View style={styles.fingersBox}>
            <Text style={styles.fingersText}>{fingers}</Text>
          </View>
        )}
        {/* Pushup position indicator */}
        {(exerciseType === 'pushups' || exerciseType === 'diamond_pushups') && (pushupData || diamondPushupData) && (
          <View style={[styles.fingersBox, { top: 70, right: 16 }]}>
            <Text style={styles.fingersText}>
              {exerciseType === 'diamond_pushups' && diamondPushupData 
                ? (diamondPushupData.is_down ? 'DOWN' : diamondPushupData.is_up ? 'UP' : 'MID')
                : (pushupData?.is_down ? 'DOWN' : pushupData?.is_up ? 'UP' : 'MID')}
            </Text>
          </View>
        )}

        {previewSize && landmarks && (
          <Canvas style={StyleSheet.absoluteFill}>
            {/* shoulders, hips */}
            {seg(L?.l_shoulder, L?.r_shoulder, 'shoulders', '#80deea')}
            {seg(L?.l_hip, L?.r_hip, 'hips', '#80cbc4')}

            {/* torso: nose -> mid-shoulders -> mid-hips */}
            {(() => {
              const ms = L?.l_shoulder && L?.r_shoulder ? { x: (L.l_shoulder.x + L.r_shoulder.x) / 2, y: (L.l_shoulder.y + L.r_shoulder.y) / 2 } : undefined;
              const mh = L?.l_hip && L?.r_hip ? { x: (L.l_hip.x + L.r_hip.x) / 2, y: (L.l_hip.y + L.r_hip.y) / 2 } : undefined;
              return [
                seg(L?.nose, ms, 'nose-shoulders', '#ffab91'),
                seg(ms, mh, 'shoulders-hips', '#ffcc80'),
              ];
            })()}

            {/* arms */}
            {seg(L?.l_shoulder, L?.l_wrist, 'l-arm', '#4dd0e1')}
            {seg(L?.r_shoulder, L?.r_wrist, 'r-arm', '#4dd0e1')}
            {/* elbows for pushups */}
            {exerciseType === 'pushups' && (
              <>
                {seg(L?.l_shoulder, L?.l_elbow, 'l-upper-arm', '#4dd0e1')}
                {seg(L?.l_elbow, L?.l_wrist, 'l-lower-arm', '#4dd0e1')}
                {seg(L?.r_shoulder, L?.r_elbow, 'r-upper-arm', '#4dd0e1')}
                {seg(L?.r_elbow, L?.r_wrist, 'r-lower-arm', '#4dd0e1')}
              </>
            )}

            {/* legs */}
            {seg(L?.l_hip, L?.l_ankle, 'l-leg', '#ffd54f')}
            {seg(L?.r_hip, L?.r_ankle, 'r-leg', '#ffd54f')}

            {/* dots */}
            {dot(L?.nose, 'nose')}
            {dot(L?.l_shoulder, 'ls')}
            {dot(L?.r_shoulder, 'rs')}
            {dot(L?.l_hip, 'lh')}
            {dot(L?.r_hip, 'rh')}
            {dot(L?.l_wrist, 'lw')}
            {dot(L?.r_wrist, 'rw')}
            {dot(L?.l_ankle, 'la')}
            {dot(L?.r_ankle, 'ra')}
            {/* elbows for pushups */}
            {exerciseType === 'pushups' && (
              <>
                {dot(L?.l_elbow, 'le')}
                {dot(L?.r_elbow, 're')}
              </>
            )}
          </Canvas>
        )}
      </View>
    );
  }, [reps, phase, landmarks, previewSize, device, serverFps, feetApart, exerciseType, pushupData, fingers, wsConnected, msgCount]);

  if (!device) {
    return (
      <View style={styles.center}>
        <ActivityIndicator />
      </View>
    );
  }

  if (!hasPermission) {
    return (
      <View style={styles.center}>
        <Text>Camera permission is required</Text>
      </View>
    );
  }

  return (
    <View
      style={styles.container}
      onLayout={(e) => setPreviewSize({ width: e.nativeEvent.layout.width, height: e.nativeEvent.layout.height })}
    >
      <StatusBar barStyle={'light-content'} />
      <Camera
        style={StyleSheet.absoluteFill}
        device={device}
        isActive={true}
        photo={true}
        ref={cameraRef}
      />
      {overlay}
      {/* Add a button to toggle exercise type in live tracking mode */}
      <TouchableOpacity 
        onPress={toggleExerciseType} 
        style={{
          position: 'absolute',
          bottom: 50,
          alignSelf: 'center',
          backgroundColor: exerciseType === 'pushups' ? '#2196F3' : '#4CAF50',
          paddingVertical: 12,
          paddingHorizontal: 20,
          borderRadius: 8,
          zIndex: 1000
        }}
      >
        <Text style={{ color: 'white', fontSize: 16 }}>
          {exerciseType === 'pushups' ? 'Pushups Mode' : 'Jumping Jacks Mode'}
        </Text>
      </TouchableOpacity>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: 'black' },
  center: { flex: 1, alignItems: 'center', justifyContent: 'center' },
  overlayBox: { position: 'absolute', top: 16, left: 16, backgroundColor: 'rgba(0,0,0,0.5)', paddingHorizontal: 12, paddingVertical: 8, borderRadius: 8 },
  overlayText: { color: 'white', fontSize: 18, fontWeight: '600' },
  overlaySub: { color: '#cfd8dc', fontSize: 12, marginTop: 2 },
  fingersBox: { position: 'absolute', top: 16, right: 16, backgroundColor: 'rgba(0,0,0,0.5)', paddingHorizontal: 12, paddingVertical: 8, borderRadius: 8, alignItems: 'center', justifyContent: 'center' },
  fingersText: { color: '#ffeb3b', fontSize: 36, fontWeight: '800' },
});
