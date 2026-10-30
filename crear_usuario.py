from passlib.context import CryptContext
from database import escribir, leer

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def agregar_usuario_seguro():
    print("=========================================")
    print("       CREAR NUEVO USUARIO - KOVAQ       ")
    print("=========================================")
    
    # 1. Verificar los roles existentes en la base de datos
    roles = leer("SELECT id, nombre FROM roles")
    if not roles:
        print("\n❌ Error: La tabla 'roles' está vacía.")
        print("💡 Solución: Enciende tu servidor primero con 'uvicorn main:app --reload' para que la base de datos cree los roles automáticamente.")
        return
        
    print("\nRoles disponibles en tu base de datos:")
    for r in roles:
        print(f"  [ID: {r[0]}] -> {r[1]}")
        
    try:
        rol_id_elegido = int(input("\nIngresa el ID del rol que deseas asignar: "))
    except ValueError:
        print("❌ Debes ingresar un número válido.")
        return

    email = input("Ingresa el correo del usuario: ").strip()
    password = input("Ingresa la contraseña: ").strip()
    nombre = input("Nombre completo: ").strip()

    if not email or not password:
        print("❌ El correo y la contraseña no pueden estar vacíos.")
        return

    # 2. Encriptar contraseña con Bcrypt
    hashed_pwd = pwd_context.hash(password)

    try:
        # 3. Guardar en la base de datos usando el rol válido elegido
        escribir(
            "INSERT INTO usuarios (username, password_hash, nombre_completo, rol_id) VALUES (?, ?, ?, ?)",
            (email, hashed_pwd, nombre if nombre else "Usuario", rol_id_elegido)
        )
        print(f"\n✅ ¡Usuario '{email}' creado y encriptado con éxito!")
    except Exception as e:
        print(f"\n❌ Error al guardar en la base de datos: {e}")

if __name__ == "__main__":
    agregar_usuario_seguro()