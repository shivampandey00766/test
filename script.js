import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

function main() {
    const canvas = document.querySelector('#c');
    const fileInput = document.getElementById('file-input');
    const renderer = new THREE.WebGLRenderer({ antialias: true, canvas });

    const fov = 45;
    const aspect = 2;  // the canvas default
    const near = 0.1;
    const far = 100;
    const camera = new THREE.PerspectiveCamera(fov, aspect, near, far);
    camera.position.set(0, 0, 5);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.screenSpacePanning = false;
    controls.minDistance = 1;
    controls.maxDistance = 50;
    controls.target.set(0, 0, 0);
    controls.update();

    const scene = new THREE.Scene();
    scene.background = new THREE.Color('white');

    {
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
        scene.add(ambientLight);

        const pointLight = new THREE.PointLight(0xffffff, 0.5);
        pointLight.position.set(2, 3, 4);
        scene.add(pointLight);
    }

    let planeMesh;

    fileInput.addEventListener('change', (event) => {
        const file = event.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = (e) => {
                const textureLoader = new THREE.TextureLoader();
                textureLoader.load(e.target.result, (texture) => {
                    if (planeMesh) {
                        scene.remove(planeMesh);
                        planeMesh.geometry.dispose();
                        if (planeMesh.material.map) {
                            planeMesh.material.map.dispose();
                        }
                        planeMesh.material.dispose();
                    }

                    const imageAspect = texture.image.width / texture.image.height;
                    const geometry = new THREE.PlaneGeometry(1, 1);
                    const material = new THREE.MeshBasicMaterial({ map: texture });
                    planeMesh = new THREE.Mesh(geometry, material);

                    const planeHeight = 4;
                    const planeWidth = planeHeight * imageAspect;
                    planeMesh.scale.set(planeWidth, planeHeight, 1);

                    scene.add(planeMesh);
                });
            };
            reader.readAsDataURL(file);
        }
    });

    function resizeRendererToDisplaySize(renderer) {
        const canvas = renderer.domElement;
        const width = canvas.clientWidth;
        const height = canvas.clientHeight;
        const needResize = canvas.width !== width || canvas.height !== height;
        if (needResize) {
            renderer.setSize(width, height, false);
        }
        return needResize;
    }

    function animate() {
        requestAnimationFrame(animate);

        if (resizeRendererToDisplaySize(renderer)) {
            const canvas = renderer.domElement;
            camera.aspect = canvas.clientWidth / canvas.clientHeight;
            camera.updateProjectionMatrix();
        }

        controls.update();
        renderer.render(scene, camera);
    }

    animate();
}

main();
