package Model;

import java.util.ArrayList;

import Observer.Observer;

public class Hero {
	
	private String name;
	private ArrayList<Observer> observerList = new ArrayList<>();
	
	public Hero(String name) {
		this.name = name;
	}
	
	public void addObserver(Observer o) {
		observerList.add(o);
	}
	
	public void removeObserver(Observer o) {
		observerList.remove(o);
	}
	
	private void notifyObserver(String message) {
		for (Observer o : observerList) {
			o.update(name + " : " + message);
		}
	}
	
	public void kill() {
		notifyObserver("Enemy Killed");
	}
	
	public void die() {
		notifyObserver("Team is dead");
	}
	
	public void getTurtle() {
		notifyObserver("Team got Turt;e");
	}
	
	
	
}
