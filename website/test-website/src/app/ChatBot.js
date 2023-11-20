// import { useState } from "react";

// const ChatbotComponent = () => {
//   const [userMessage, setUserMessage] = useState("");
//   const [chatHistory, setChatHistory] = useState([]);

//   const sendMessage = async () => {
//     const response = await fetch("/api/chat", {
//       method: "POST",
//       headers: {
//         "Content-Type": "application/json",
//       },
//       body: JSON.stringify({ user_message: userMessage }),
//     });

//     const data = await response.json();
//     setChatHistory([...chatHistory, { user: userMessage, bot: data.response }]);
//     setUserMessage("");
//   };

//   return (
//     <div>
//       <div>
//         {chatHistory.map((message, index) => (
//           <div key={index}>
//             <strong>User:</strong> {message.user}
//             <br />
//             <strong>Bot:</strong> {message.bot}
//             <hr />
//           </div>
//         ))}
//       </div>
//       <input
//         type="text"
//         value={userMessage}
//         onChange={(e) => setUserMessage(e.target.value)}
//       />
//       <button onClick={sendMessage}>Send</button>
//     </div>
//   );
// };

// export default ChatbotComponent;
