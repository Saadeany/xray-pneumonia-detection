# Use a lightweight Node image
FROM node:20-alpine

# Set the working directory
WORKDIR /app

# Copy package files and install dependencies
COPY package.json package-lock.json* ./
RUN npm install

# Copy the rest of the frontend code
COPY . .

# Expose the Vite development port
EXPOSE 5173

# Start the Vite development server (host ensures it works in Docker)
CMD ["npm", "run", "dev", "--", "--host"]