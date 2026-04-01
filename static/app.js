document.addEventListener("DOMContentLoaded", () => {
    // Initialize Map explicitly bound to high canvas limits
    const map = L.map('map', {
        preferCanvas: true 
    }).setView([40.764, -111.902], 15);

    // Mapbox/CartoDB Dark Matter strictly loads tiles, no complex vectors needed
    L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OSM</a> contributors',
        maxZoom: 20
    }).addTo(map);

    const redIcon = new L.Icon({
        iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-red.png',
        shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/0.7.7/images/marker-shadow.png',
        iconSize: [25, 41],
        iconAnchor: [12, 41],
        popupAnchor: [1, -34],
        shadowSize: [41, 41]
    });

    // Ingest the local Flask API topology
    fetch("/api/racks")
        .then(response => response.json())
        .then(data => {
            const stats = document.getElementById("stats-pill");
            
            if (!data || data.length === 0) {
                stats.textContent = "No detections logged.";
                return;
            }

            stats.textContent = `${data.length} Positive Detections Mapped`;
            const group = L.featureGroup();

            // Mathematically plot each object vector array
            data.forEach(record => {
                const lat = parseFloat(record.latitude);
                const lon = parseFloat(record.longitude);
                if (isNaN(lat) || isNaN(lon)) return;

                const score = parseFloat(record.confidence || 0).toFixed(2);
                
                // Securely encode absolute path directly back into backend API to bypass cross-origin browser filesystem locks
                const imageRoute = `/api/image?path=${encodeURIComponent(record.image_path)}`;

                const popupHtml = `
                    <div class="custom-popup">
                        <h3>Hit: ${record.image_id}</h3>
                        <p><strong>Model:</strong> ${record.model_name}</p>
                        <p><strong>Confidence:</strong> ${score}</p>
                        <p><strong>Captured:</strong> ${record.captured_at || 'Unknown'}</p>
                        <!-- Lazy loading stops memory hangs if the user dumps thousands of boxes -->
                        <img src="${imageRoute}" alt="Bounding Box Evidence" loading="lazy">
                    </div>
                `;

                const marker = L.marker([lat, lon], {icon: redIcon})
                 .bindPopup(popupHtml);
                group.addLayer(marker);
            });
            
            group.addTo(map);
            
            // Instantly snap map bounding view to encapsulate all current clusters
            if (data.length > 0) {
                map.fitBounds(group.getBounds().pad(0.1));
            }
        })
        .catch(err => {
            console.error("API Fetch Error:", err);
            document.getElementById("stats-pill").textContent = "API Interruption";
        });
});
