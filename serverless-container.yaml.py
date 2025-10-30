name: exam-scorer-api
spec:
  connectivity:
    network_id: default
  containers:
    - name: api
      image: cr.yandex/your-registry-id/exam-scorer:latest
      command:
        - python
        - -m
        - uvicorn
        - app.main:api
        - --host
        - 0.0.0.0
        - --port
        - "8000"
      ports:
        - containerPort: 8000
        protocol: TCP
      resources:
        memory: "2048MB"
        cores: "1"
      probes:
        http:
          path: /health
          port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5