const cam = document.getElementById('cam');

const startVideo = () => {
    navigator.mediaDevices.enumerateDevices()
        .then(devices => {
            if (Array.isArray(devices)) {
                devices.forEach(device => {
                    if (device.kind === 'videoinput') {
                        if (device.label.includes('FaceTime')) {
                            navigator.getUserMedia(
                                { video: { deviceId: device.deviceId } },
                                stream => cam.srcObject = stream,
                                error => console.error(error)
                            );
                        }
                    }
                });
            }
        });
};

const loadLabels = () => {
    const labels = ['Thiago Lopes', 'Giovani'];
    return Promise.all(labels.map(async label => {
        const descriptions = [];
        for (let i = 1; i <= 1; i++) {
            const img = await faceapi.fetchImage(`/assets/lib/face-api/labels/${label}/${i}.jpg`);
            const detections = await faceapi
                .detectSingleFace(img)
                .withFaceLandmarks()
                .withFaceDescriptor();
            descriptions.push(detections.descriptor);
        }
        return new faceapi.LabeledFaceDescriptors(label, descriptions);
    }));
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

    let lastDetectionTime = Date.now();
    const detectionInterval = 100;
    const updateInterval = 1000;

    let lastDetections = [];
    let lastResults = [];

    setInterval(async () => {
        const currentTime = Date.now();
        if (currentTime - lastDetectionTime > detectionInterval) {
            const detections = await faceapi
                .detectAllFaces(
                    cam,
                    new faceapi.TinyFaceDetectorOptions()
                )
                .withFaceLandmarks()
                .withFaceExpressions()
                .withAgeAndGender()
                .withFaceDescriptors();
            lastDetections = faceapi.resizeResults(detections, canvasSize);
            const faceMatcher = new faceapi.FaceMatcher(labels, 0.9);
            lastResults = lastDetections.map(d =>
                faceMatcher.findBestMatch(d.descriptor)
            );
            lastDetectionTime = currentTime;
        }

        canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);
        faceapi.draw.drawDetections(canvas, lastDetections);
        faceapi.draw.drawFaceLandmarks(canvas, lastDetections);
        faceapi.draw.drawFaceExpressions(canvas, lastDetections);

        if (currentTime - lastDetectionTime < detectionInterval) {
            lastDetections.forEach(detection => {
                const { age, gender, genderProbability } = detection;
                new faceapi.draw.DrawTextField([
                    `${parseInt(age, 10)} anos`,
                    `${translateGender(gender)} (${parseInt(genderProbability * 100, 10)}%)`
                ], detection.detection.box.topRight).draw(canvas);
            });

            lastResults.forEach((result, index) => {
                const box = lastDetections[index].detection.box;
                const { label, distance } = result;
                new faceapi.draw.DrawTextField([
                    `${label} (${parseInt(distance * 100, 10)}%)`
                ], box.bottomRight).draw(canvas);
            });
        }
    }, detectionInterval);
});
