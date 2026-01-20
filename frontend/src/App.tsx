import { useState } from "react";
import Cookies from "js-cookie";
import { Chat } from "./components/Chat";
import { AuthModal } from "./components/AuthModal";

function App() {
  const [isAuthenticated, setIsAuthenticated] = useState(
    () => Cookies.get("mentor_auth") === "true"
  );

  return (
    <>
      {!isAuthenticated && (
        <AuthModal onAuthenticated={() => setIsAuthenticated(true)} />
      )}
      <Chat />
    </>
  );
}

export default App;
