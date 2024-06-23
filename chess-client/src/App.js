import React, { useState } from 'react';
import { Chessboard } from 'react-chessboard';
import { Chess } from 'chess.js';
import './App.css';

const App = () => {
  const [game, setGame] = useState(new Chess());
  const [isUserTurn, setIsUserTurn] = useState(true);
  const [moveHistory, setMoveHistory] = useState([]);
  const [gameOver, setGameOver] = useState(false);
  const [result, setResult] = useState('');
  const [isFetching, setIsFetching] = useState(false);
  const [currentMoveIndex, setCurrentMoveIndex] = useState(0);

  const handleGameOver = (gameCopy) => {
    setGameOver(true);
    console.log(gameCopy.turn())
    if (gameCopy.isCheckmate()) {
      setResult(gameCopy.turn() === 'b' ? 'White wins by checkmate' : 'Black wins by checkmate');
    } else if (gameCopy.isDraw()) {
      setResult('Draw');
    } else if (gameCopy.isStalemate()) {
      setResult('Draw by stalemate');
    } else if (gameCopy.isInsufficientMaterial()) {
      setResult('Draw by insufficient material');
    } else if (gameCopy.isThreefoldRepetition()) {
      setResult('Draw by threefold repetition');
    } else if (gameCopy.halfmoves() >= 50) {
      setResult('Draw by 50-move rule');
    } else {
      setResult('Game over');
    }
  };

  const makeAMove = async (move) => {
    const gameCopy = new Chess(game.fen());

    let userMove;
    try {
      userMove = gameCopy.move(move);
    } catch (error) {
      setIsUserTurn(true);
      return false;
    }

    const newMoveHistory = [...moveHistory];
    if (newMoveHistory.length === 0 || newMoveHistory[newMoveHistory.length - 1].length === 2) {
      newMoveHistory.push([userMove.san]);
    } else {
      newMoveHistory[newMoveHistory.length - 1].push(userMove.san);
    }
    setMoveHistory(newMoveHistory);
    setCurrentMoveIndex(newMoveHistory.length);


    if (gameCopy.isGameOver()) {
      setGame(gameCopy);
      setIsUserTurn(false);
      console.log("game is over apparently and white should win.", gameCopy, isUserTurn)
      handleGameOver(gameCopy);
      return true;
    }

    setGame(gameCopy);
    setIsUserTurn(false);
    setIsFetching(true);

    const response = await fetch('http://localhost:8000/move/', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ fen: gameCopy.fen() }),
    });

    if (!response.ok) {
      console.error('Network response was not ok');
      setIsUserTurn(true);
      setIsFetching(false);
      return false;
    }

    const data = await response.json();
    const bestMove = data.move;

    const botMove = gameCopy.move(bestMove);
    newMoveHistory[newMoveHistory.length - 1].push(botMove.san);
    setMoveHistory(newMoveHistory);
    setCurrentMoveIndex(newMoveHistory.length);

    if (gameCopy.isGameOver()) {
      console.log("game is over apparently and black should win.", gameCopy, isUserTurn)
      setIsFetching(false);
      handleGameOver(gameCopy);
      return true;
    }

    setGame(gameCopy)
    setIsUserTurn(true);
    setIsFetching(false);
    return true;

  };

  const onDrop = async (sourceSquare, targetSquare, piece) => {
    if (!isUserTurn) return false;

    const moveMade = await makeAMove({
      from: sourceSquare,
      to: targetSquare,
      promotion: piece.slice(-1).toLowerCase(),
    });

    return moveMade;
  };

  const restartGame = () => {
    setGame(new Chess());
    setIsUserTurn(true);
    setMoveHistory([]);
    setCurrentMoveIndex(0);
    setGameOver(false);
    setResult('');
  };

  const undoMove = () => {
    if (currentMoveIndex === 0) return;
    const newMoveIndex = currentMoveIndex - 1;
    const newMoveHistory = moveHistory.slice(0, newMoveIndex);
    const newGame = new Chess();
    newMoveHistory.flat().forEach(move => newGame.move(move));
    setGame(newGame);
    setMoveHistory(newMoveHistory);
    setCurrentMoveIndex(newMoveIndex);
    setIsUserTurn(true);
    setGameOver(false)
    setResult('')
  };

  return (
    <div className="main">
      <div className="container">
        <Chessboard
          position={game.fen()}
          onPieceDrop={onDrop}
          areArrowsAllowed={true}
          arePiecesDraggable={!gameOver && isUserTurn}
        />
      </div>
      <div className="controls">
      </div>
      <div className="secondary-container">
        <h3>Move History</h3>
        <ul>
          {moveHistory.map((movePair, index) => (
            <li key={index}>
              {index + 1}. {movePair[0]} {movePair[1] ? movePair[1] : ''}
            </li>
          ))}
        </ul>
        <div className="actions">
          <p>{result}</p>
          <div
            className={`${isFetching ? 'fetching' : 'action-button'}`}
            onClick={() => !isFetching && restartGame()}
          >
            {isFetching ? 'Fetching' : 'Restart Game'}
          </div>
          <div
            className={`${isFetching || currentMoveIndex === 0 ? 'fetching' : 'action-button'}`}
            onClick={() => !isFetching && undoMove()}
          >
            {isFetching ? 'Fetching' : 'Undo'}
          </div>
        </div>
      </div>
    </div>


  );
};

export default App;
