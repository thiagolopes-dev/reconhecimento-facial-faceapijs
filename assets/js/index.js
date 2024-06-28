const cam = document.getElementById('cam');

const startVideo = () => {
    navigator.mediaDevices.enumerateDevices()
        .then(devices => {
            if (Array.isArray(devices)) {
                devices.forEach(device => {
                    if (device.kind === 'videoinput') {
                        if (device.label.includes('FaceTime')) {
                            navigator.mediaDevices.getUserMedia(
                                { video: { deviceId: device.deviceId } }
                            ).then(stream => cam.srcObject = stream)
                            .catch(error => console.error(error));
                        }
                    }
                });
            }
            console.log(devices);
        });
};

const loadLabels = async () => {
    const labels = ['Thiago Lopes', 'Giovani'];
    return Promise.all(labels.map(async label => {
        const descriptions = [];
        // Aumente o número de imagens por pessoa para melhorar a precisão
        for (let i = 1; i <= 5; i++) {
            try {
                const img = await faceapi.fetchImage(`/assets/lib/face-api/labels/${label}/${i}.jpg`);
                const detections = await faceapi
                    .detectSingleFace(img)
                    .withFaceLandmarks()
                    .withFaceDescriptor();
                if (detections) {
                    descriptions.push(detections.descriptor);
                } else {
                    console.warn(`No face detected for ${label} in image ${i}`);
                }
            } catch (error) {
                console.error(`Error loading image ${i} for ${label}:`, error);
            }
        }
        if (descriptions.length > 0) {
            return new faceapi.LabeledFaceDescriptors(label, descriptions);
        } else {
            console.warn(`No valid descriptors for ${label}`);
            return null;
        }
    })).then(descriptors => descriptors.filter(d => d !== null)); // Filter out null descriptors
};

Promise.all([
    faceapi.nets.tinyFaceDetector.loadFromUri('/assets/lib/face-api/models'),
    faceapi.nets.faceLandmark68Net.loadFromUri('/assets/lib/face-api/models'),
    faceapi.nets.faceRecognitionNet.loadFromUri('/assets/lib/face-api/models'),
    faceapi.nets.faceExpressionNet.loadFromUri('/assets/lib/face-api/models'),
    faceapi.nets.ageGenderNet.loadFromUri('/assets/lib/face-api/models'),
    faceapi.nets.ssdMobilenetv1.loadFromUri('/assets/lib/face-api/models'),
]).then(startVideo);

const translateGender = (gender) => {
    return gender === 'male' ? 'masculino' : 'feminino';
};

cam.addEventListener('play', async () => {
    const canvas = faceapi.createCanvasFromMedia(cam);
    const canvasSize = {
        width: cam.width,
        height: cam.height
    };
    const labels = await loadLabels();
    faceapi.matchDimensions(canvas, canvasSize);
    document.body.appendChild(canvas);

    // Aumente a precisão do FaceMatcher ajustando o limiar
    const faceMatcher = new faceapi.FaceMatcher(labels, 0.6);
    
    const detectionInterval = 100;

    setInterval(async () => {
        const detections = await faceapi
            .detectAllFaces(
                cam,
                new faceapi.TinyFaceDetectorOptions()
            )
            .withFaceLandmarks()
            .withFaceExpressions()
            .withAgeAndGender()
            .withFaceDescriptors();
        
        const resizedDetections = faceapi.resizeResults(detections, canvasSize);
        
        const results = resizedDetections.map(d =>
            faceMatcher.findBestMatch(d.descriptor)
        );

        canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);
        faceapi.draw.drawDetections(canvas, resizedDetections);
        faceapi.draw.drawFaceLandmarks(canvas, resizedDetections);
        faceapi.draw.drawFaceExpressions(canvas, resizedDetections);

        resizedDetections.forEach((detection, i) => {
            const { age, gender, genderProbability } = detection;
            const box = detection.detection.box;
            const drawBox = new faceapi.draw.DrawBox(box, { label: `${parseInt(age, 10)} anos, ${translateGender(gender)} (${parseInt(genderProbability * 100, 10)}%)` });
            drawBox.draw(canvas);
            
            const result = results[i];
            if (result.label !== 'unknown') {  // Verifique se o rótulo é 'unknown' em vez de 'Desconhecido'
                new faceapi.draw.DrawTextField([
                    `${result.label} (${parseInt(result.distance * 100, 10)}%)`
                ], box.bottomRight).draw(canvas);
            }
        });
    }, detectionInterval);
});
