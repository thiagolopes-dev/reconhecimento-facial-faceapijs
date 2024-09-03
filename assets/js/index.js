const cam = document.getElementById('cam');

const startVideo = () => {
    // Solicita permissão para acessar a câmera
    navigator.mediaDevices.getUserMedia({ video: true })
        .then(stream => {
            cam.srcObject = stream;
            enumerateDevices();
        })
        .catch(error => {
            console.error('Erro ao acessar a câmera:', error);
            alert('Permissão para usar a câmera negada ou não disponível.');
        });
};

const enumerateDevices = () => {
    navigator.mediaDevices.enumerateDevices()
        .then(devices => {
            const videoDevices = devices.filter(device => device.kind === 'videoinput');
            
            // Se houver mais de uma câmera, peça para o usuário selecionar
            if (videoDevices.length > 1) {
                let options = "Escolha a câmera:\n";
                videoDevices.forEach((device, index) => {
                    options += `${index + 1}: ${device.label}\n`;
                });
                const selection = prompt(options);
                const selectedDevice = videoDevices[parseInt(selection, 10) - 1];
                
                if (selectedDevice) {
                    // Usa a câmera selecionada pelo usuário
                    navigator.mediaDevices.getUserMedia({
                        video: { deviceId: selectedDevice.deviceId }
                    }).then(stream => cam.srcObject = stream)
                    .catch(error => console.error('Erro ao acessar a câmera selecionada:', error));
                } else {
                    alert('Seleção inválida. Usando a câmera padrão.');
                    startVideo();
                }
            }
        })
        .catch(error => console.error('Erro ao enumerar dispositivos:', error));
};

// Função para carregar os rótulos (labels) com os nomes das pessoas
const loadLabels = async () => {
    const labels = ['Thiago Lopes', 'Micheletti', 'Argati', 'Rafael Cita', 'Da Costa', 'Michele'];
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

// Carrega os modelos do FaceAPI e inicia o vídeo
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
        const results = resizedDetections.map(d => faceMatcher.findBestMatch(d.descriptor));

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
            if (result.label !== 'Desconhecido') {
                new faceapi.draw.DrawTextField([
                    `${result.label} (${parseInt(result.distance * 100, 10)}%)`
                ], box.bottomRight).draw(canvas);
            }
        });
    }, detectionInterval);
});
