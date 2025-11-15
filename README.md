# GoopLum Backend

## 🚀 Quick Start

### Prerequisites
- Python 3.12+
- [UV](https://docs.astral.sh/uv/) for package management
- [Supabase](https://supabase.com/) account for database

### Database Setup

1. **Create a new Supabase project:**
   - Go to [supabase.com](https://supabase.com/)
   - Click "New Project"
   - Choose your organization
   - Set project name (e.g., "gooplum")
   - Set a strong database password
   - Choose your region
   - Click "Create new project"

2. **Set up database tables:**
   - Open your Supabase project dashboard
   - Go to the **SQL Editor**
   - Click "New query"
   - Copy and paste the contents of `database_setup.sql`
   - Click "Run" to execute the setup script

3. **Get your Supabase credentials:**
   - In your Supabase project, go to **Settings** → **API**
   - Copy the **Project URL** and **anon public** key

4. **Configure environment variables:**
   - Create a `.env` file in the backend directory:
   ```bash
   # Supabase Configuration
   SUPABASE_URL=https://your-project-id.supabase.co
   SUPABASE_ANON_KEY=your-anon-key-here
   ```

### Local Setup

```bash
# Install dependencies
uv sync

# Install development dependencies
uv sync --dev
```

### Running the Server

```bash
# Development mode (recommended)
uv run langgraph dev
```

The server will start on `http://localhost:2024`

## 🛠️ Development Commands

```bash
# Code formatting
uv run black .

# Linting and type checking
uv run ruff check .
```

## 📚 API Documentation

Once the server is running, you can access the API Docs at `http://localhost:2024/docs`
